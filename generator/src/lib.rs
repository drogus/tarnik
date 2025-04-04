use anyhow::anyhow;
use indexmap::IndexMap;
use proc_macro::TokenStream;
use regex::bytes;
use std::{ops::Deref, str::FromStr};
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
    token::{self, Brace, Comma, PathSep, Semi},
    AngleBracketedGenericArguments, Attribute, Expr, ExprBinary, ExprClosure, ExprForLoop,
    ExprUnary, GenericArgument, Lit, LitStr, Local, Meta, Pat, PatType, PathArguments, Type,
};
use utf16string::{LittleEndian, WString, LE};

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::{Literal, Punct, Spacing, Span, TokenStream as TokenStream2, TokenTree};
use quote::{quote, ToTokens};
use syn::{braced, parenthesized, parse_macro_input, Ident, Result, Stmt, Token};

use tarnik_ast::{
    Global, InstructionsList, Signature, StructField, TypeDefinition, WasmType, WatFunction,
    WatInstruction, WatModule,
};

#[derive(Debug, Clone)]
enum BodyElement {
    Statement(Stmt),
    TryCatch {
        r#try: Vec<BodyElement>,
        catches: Vec<(Ident, Vec<Parameter>, Vec<BodyElement>)>,
        catch_all: Option<Vec<BodyElement>>,
    },
    If {
        condition: Expr,
        body: Vec<BodyElement>,
        r#else: Option<Vec<BodyElement>>,
    },
    While {
        condition: Expr,
        body: Vec<BodyElement>,
    },
}

#[derive(Debug, Clone)]
struct Parameter {
    name: String,
    ty: WasmType,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    parameters: Vec<Parameter>,
    return_type: Option<WasmType>,
    body: Vec<BodyElement>,
}

#[derive(Debug, Clone)]
struct GlobalScope {
    module: WatModule,
    functions: Vec<Function>,
}

impl GlobalScope {
    fn parse_function(
        &mut self,
        input: &mut ParseStream,
        import_next: &mut Option<(String, String)>,
        export_next: &mut Option<String>,
    ) -> Result<()> {
        let function = parse_function(input)?;

        if let Some(export_name) = export_next {
            self.module
                .add_export(&*export_name, "func", format!("${}", function.name));
            *export_next = None;
        } else if let Some((namespace, name)) = import_next {
            self.module.add_import(
                &*namespace,
                &*name,
                WasmType::func(
                    Some(format!("${}", function.name)),
                    function
                        .parameters
                        .iter()
                        .map(|p| (Some(p.name.clone()), p.ty.clone()))
                        .collect(),
                    function.return_type.clone(),
                ),
            );

            *import_next = None;
        }

        let wat_function = self.function_to_wat_function_signature(&function)?;
        self.module.add_function(wat_function);
        self.functions.push(function);

        Ok(())
    }

    fn parse_type(
        &mut self,
        input: &mut ParseStream,
        export_next: &mut Option<String>,
    ) -> Result<()> {
        let (name, ty) = parse_type_def(input)?;
        self.module
            .types
            .push(TypeDefinition::Type(name.clone(), ty));
        if let Some(export_name) = export_next {
            self.module
                .add_export(&*export_name, "type", format!("${}", name));
            *export_next = None;
        }
        // TODO: handle importing types too
        Ok(())
    }

    fn parse_struct(
        &mut self,
        mut input: &mut ParseStream,
        export_next: &mut Option<String>,
    ) -> Result<()> {
        let (name, ty) = parse_struct(&mut input)?;
        self.module
            .types
            .push(TypeDefinition::Type(format!("${name}"), ty));
        if let Some(export_name) = export_next {
            self.module
                .add_export(&*export_name, "type", format!("${}", name));
            *export_next = None;
        }

        Ok(())
    }

    fn parse_attr_macro(
        &mut self,
        input: &mut ParseStream,
        import_next: &mut Option<(String, String)>,
        export_next: &mut Option<String>,
    ) -> Result<()> {
        let attrs = input.call(Attribute::parse_outer)?;
        if attrs.len() > 1 {
            return Err(syn::Error::new_spanned(
                attrs[1].clone(),
                "Only one macro attribute per item is supported at the moment",
            ));
        }

        let ident = &attrs[0].meta.path().segments[0].ident;
        let name = ident.to_string();
        if name == "export" {
            if let Meta::List(list) = &attrs[0].meta {
                let tokens = &list.tokens;
                let str: Literal = syn::parse2(tokens.clone())?;
                // TODO: is there a better way to extract it?
                *export_next = Some(str.to_string().trim_matches('"').to_string());
            } else {
                return Err(syn::Error::new_spanned(
                    &attrs[0].meta,
                    "Export macro accepts only a string literal as an argument",
                ));
            }
        } else if name == "import" {
            if let Meta::List(list) = &attrs[0].meta {
                let tokens = list.tokens.clone();

                let literals = ::syn::parse::Parser::parse2(
                    Punctuated::<Literal, Token![,]>::parse_terminated,
                    tokens,
                )?;

                // TODO: is there a better way to extract it?
                *import_next = Some((
                    literals[0].to_string().trim_matches('"').to_string(),
                    literals[1].to_string().trim_matches('"').to_string(),
                ));
            } else {
                return Err(syn::Error::new_spanned(
                    &attrs[0].meta,
                    "Export macro accepts only a string literal as an argument",
                ));
            }
        } else {
            return Err(syn::Error::new_spanned(
                ident,
                "{name} attribute macro doesn't exist",
            ));
        }

        Ok(())
    }

    fn parse_macro(
        &mut self,
        input: &mut ParseStream,
        export_next: &mut Option<String>,
    ) -> Result<()> {
        let mcro: syn::Macro = input.parse()?;
        let macro_name = &mcro.path.segments[0].ident.to_string();
        match macro_name.as_str() {
            "memory" => {
                let expressions = ::syn::parse::Parser::parse2(
                    Punctuated::<Expr, Token![,]>::parse_terminated,
                    mcro.tokens.clone(),
                )?;

                if expressions.len() < 2 || expressions.len() > 3 {
                    return Err(syn::Error::new_spanned(
                        mcro,
                        "memory!() must have 2 or 3 arguments: name, size and optional max_size",
                    ));
                }

                let name_expr = &expressions[0];
                let size_expr = &expressions[1];
                let max_size_expr = &expressions.get(2);

                let name = if let Expr::Lit(expr_lit) = name_expr {
                    if let Lit::Str(str) = &expr_lit.lit {
                        str.value()
                    } else {
                        return Err(syn::Error::new_spanned(
                            mcro,
                            "first argument to memory!() has to be a string literal",
                        ));
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        mcro,
                        "first argument to memory!() has to be a string literal",
                    ));
                };

                let size: i32 = if let Expr::Lit(expr_lit) = size_expr {
                    if let Lit::Int(int) = &expr_lit.lit {
                        int.base10_parse()?
                    } else {
                        return Err(syn::Error::new_spanned(
                            mcro,
                            "Second argument to memory!() has to be an integer",
                        ));
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        mcro,
                        "Second argument to memory!() has to be an integer",
                    ));
                };

                let max_size: Option<i32> = if let Some(max_size_expr) = max_size_expr {
                    if let Expr::Lit(expr_lit) = max_size_expr {
                        if let Lit::Int(int) = &expr_lit.lit {
                            Some(int.base10_parse()?)
                        } else {
                            return Err(syn::Error::new_spanned(
                                mcro,
                                "Third argument to memory!() has to be an integer",
                            ));
                        }
                    } else {
                        return Err(syn::Error::new_spanned(
                            mcro,
                            "Third argument to memory!() has to be an integer",
                        ));
                    }
                } else {
                    None
                };

                let label = format!("${name}");
                self.module.add_memory(&label, size, max_size);
                if let Some(export_name) = export_next {
                    self.module.add_export(&*export_name, "memory", label);
                    *export_next = None;
                }
            }
            "tag" => {
                let idents = ::syn::parse::Parser::parse2(
                    Punctuated::<Ident, Token![,]>::parse_terminated,
                    mcro.tokens.clone(),
                )?;

                if idents.len() != 2 {
                    return Err(syn::Error::new_spanned(
                                mcro.tokens,
                                "the tag! macro expects two identifiers: the name of the tag and the name of the type",
                            ));
                }

                let tag_name = format!("${}", idents[0]);
                let type_name = format!("${}", idents[1]);
                self.module.tags.insert(tag_name.clone(), type_name);
                if let Some(export_name) = export_next {
                    self.module.add_export(&*export_name, "tag", tag_name);
                    *export_next = None;
                }
            }
            "rec" => {
                ::syn::parse::Parser::parse2(
                    |mut input: ParseStream<'_>| {
                        let mut types = Vec::new();
                        while !input.is_empty() {
                            if input.peek(Token![type]) {
                                let (name, ty) = parse_type_def(&mut input)?;
                                types.push((name.to_string(), ty));
                            } else {
                                let (name, ty) = parse_struct(&mut input)?;
                                types.push((format!("${name}"), ty));
                            }
                        }

                        self.module.types.push(TypeDefinition::Rec(types));

                        Ok(())
                    },
                    mcro.tokens.clone(),
                )?;
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    macro_name,
                    format!("Top level macro {macro_name} not found"),
                ));
            }
        }

        if input.peek(token::Semi) {
            let _: token::Semi = input.parse()?;
        }

        Ok(())
    }

    fn parse_global(&mut self, mut input: &mut ParseStream) -> Result<()> {
        let _: Token![static] = input.parse()?;

        let mutable = if input.peek(Token![mut]) {
            let _: Token![mut] = input.parse()?;
            true
        } else {
            false
        };

        let ident: Ident = input.parse()?;

        let _: Token![:] = input.parse()?;

        let ty = parse_type(input)?;

        let _: Token![=] = input.parse()?;

        let expr: Expr = input.parse()?;

        let _: Token![;] = input.parse()?;

        let mut init = Vec::new();
        translate_expression(
            &mut self.module,
            &mut WatFunction::new("dummy"),
            &mut init,
            &expr,
            None,
            Some(&ty),
        )?;

        let name = format!("${}", ident);

        self.module.globals.insert(
            name.clone(),
            Global {
                name,
                ty,
                init,
                mutable,
            },
        );

        Ok(())
    }

    fn parse_global_statements(&mut self, input: &mut ParseStream) -> Result<()> {
        let mut import_next: Option<(String, String)> = None;
        let mut export_next: Option<String> = None;

        while !input.is_empty() {
            if input.peek(Token![fn]) {
                self.parse_function(input, &mut import_next, &mut export_next)?;
            } else if input.peek(Token![type]) {
                self.parse_type(input, &mut export_next)?;
            } else if input.peek(Token![struct]) {
                self.parse_struct(input, &mut export_next)?;
            } else if input.peek(Token![#]) {
                self.parse_attr_macro(input, &mut import_next, &mut export_next)?;
            } else if input.peek(syn::Ident) && input.peek2(Token![!]) {
                self.parse_macro(input, &mut export_next)?;
            } else if input.peek(Token![static]) {
                self.parse_global(input)?;
            } else {
                let attr = input.call(Attribute::parse_outer)?;
                println!("attr: {attr:#?}");
                println!("input: {input:#?}");
                todo!("other input type")
            }
        }

        Ok(())
    }

    fn function_to_wat(&mut self, function: &Function) -> Result<WatFunction> {
        let mut wat_function = WatFunction::new(&function.name);
        for param in &function.parameters {
            wat_function.add_param(format!("${}", param.name), &param.ty);
        }

        wat_function.results = if let Some(result) = &function.return_type {
            vec![result.clone()]
        } else {
            vec![]
        };

        let mut instructions = Vec::new();
        for stmt in &function.body {
            translate_body_element(&mut self.module, &mut wat_function, &mut instructions, stmt)?;
        }
        wat_function.set_body(instructions);
        Ok(wat_function)
    }

    fn function_to_wat_function_signature(&mut self, function: &Function) -> Result<WatFunction> {
        let mut wat_function = WatFunction::new(&function.name);
        for param in &function.parameters {
            wat_function.add_param(format!("${}", param.name), &param.ty);
        }

        wat_function.results = if let Some(result) = &function.return_type {
            vec![result.clone()]
        } else {
            vec![]
        };

        Ok(wat_function)
    }
}

impl Parse for GlobalScope {
    fn parse(mut input: ParseStream) -> Result<Self> {
        let mut global_scope = Self {
            module: WatModule::new(),
            functions: Vec::new(),
        };

        global_scope.parse_global_statements(&mut input)?;

        for function in global_scope.functions.clone() {
            let wat_function = global_scope.function_to_wat(&function)?;

            let existing = global_scope
                .module
                .get_function_mut(&wat_function.name)
                .unwrap();
            existing.replace(wat_function);
        }

        Ok(global_scope)
    }
}

#[derive(Clone, Debug)]
struct Field {
    mutable: bool,

    ident: Option<Ident>,

    ty: WasmType,
}

#[derive(Clone, Debug)]
struct ItemStruct {
    ident: Ident,
    fields: Vec<Field>,
}

fn data_struct(input: ParseStream) -> Result<Vec<Field>> {
    let lookahead = input.lookahead1();

    if lookahead.peek(token::Paren) {
        let content;

        parenthesized!(content in input);
        let mut punctuated = content.parse_terminated(parse_unnamed_field, Token![,])?;
        Ok(punctuated.iter_mut().map(|f| f.clone()).collect())
    } else if lookahead.peek(token::Brace) {
        let content;

        braced!(content in input);
        let mut punctuated = content.parse_terminated(parse_named_field, Token![,])?;
        Ok(punctuated.iter_mut().map(|f| f.clone()).collect())
    } else {
        Err(lookahead.error())
    }
}

impl Parse for ItemStruct {
    fn parse(input: ParseStream) -> Result<Self> {
        let _ = input.parse::<Token![struct]>()?;
        let ident = input.parse::<Ident>()?;

        let fields = data_struct(input)?;

        Ok(ItemStruct { ident, fields })
    }
}

fn parse_named_field(mut input: ParseStream) -> Result<Field> {
    let ident: Ident = input.parse()?;

    let _: Token![:] = input.parse()?;

    let mutable = input.parse::<Token![mut]>().is_ok();

    let ty = parse_type(&mut input)?;

    Ok(Field {
        mutable,
        ident: Some(ident),
        ty,
    })
}

fn parse_unnamed_field(mut input: ParseStream) -> Result<Field> {
    let ty = parse_type(&mut input)?;
    Ok(Field {
        mutable: false,
        ident: None,
        ty,
    })
}

fn parse_struct(input: &mut ParseStream) -> Result<(String, WasmType)> {
    let s: ItemStruct = input.parse()?;
    // TODO: leave idents as idents till the last moment, so it can be tracked
    let name = s.ident.to_string();
    let mut fields = Vec::new();
    for field in s.fields {
        fields.push(StructField {
            mutable: field.mutable,
            ty: field.ty,
            name: field.ident.map(|i| i.to_string()),
        });
    }

    Ok((name, WasmType::Struct(fields)))
}

fn parse_type(input: &mut ParseStream) -> Result<WasmType> {
    if input.peek(token::Bracket) {
        // custom parsing for array types
        let content;
        bracketed!(content in input);

        let mutable = content.parse::<Token![mut]>().is_ok();
        let ty = parse_type(&mut &content)?;

        Ok(WasmType::Array {
            mutable,
            ty: Box::new(ty),
        })
    } else {
        let rust_type: Type = input.parse()?;
        // TODO: Handle errors properly
        Ok(translate_type(&rust_type)
            .ok_or(anyhow!("Could not translate type in a type definition"))
            .unwrap())
    }
}

fn parse_type_def(input: &mut ParseStream) -> Result<(String, WasmType)> {
    let _: Token![type] = input.parse()?;

    let ident: Ident = input.parse()?;

    let _: Token![=] = input.parse()?;

    let ty = parse_type(input)?;

    let _: token::Semi = input.parse()?;

    Ok((format!("${ident}"), ty))
}

fn parse_param(mut input: ParseStream) -> Result<Parameter> {
    let name: Ident = input.parse()?;
    let _: Token![:] = input.parse()?;
    let ty = parse_type(&mut input)?;
    Ok(Parameter {
        name: name.to_string(),
        ty,
    })
}

fn parse_function(input: &mut ParseStream) -> Result<Function> {
    let _: Token![fn] = input.parse()?;
    let name: Ident = input.parse()?;

    let content;
    parenthesized!(content in input);

    let mut parameters = Vec::new();
    while !content.is_empty() {
        let name: Ident = content.parse()?;
        let _: Token![:] = content.parse()?;
        let ty = parse_type(&mut &content)?;

        parameters.push(Parameter {
            name: name.to_string(),
            ty,
        });

        if !content.is_empty() {
            let _: Token![,] = content.parse()?;
        }
    }

    let return_type = if input.peek(Token![->]) {
        let _: Token![->] = input.parse()?;
        Some(parse_type(input)?)
    } else {
        None
    };
    let mut body: Vec<BodyElement> = Vec::new();

    if input.peek(Brace) {
        parse_code_block(input, &mut body)?;
    } else {
        let _: Token![;] = input.parse()?;
    }

    let f = Function {
        name: name.to_string(),
        parameters,
        return_type,
        body,
    };

    Ok(f)
}

fn parse_code_block(input: &mut ParseStream, body: &mut Vec<BodyElement>) -> Result<()> {
    let content;
    braced!(content in input);
    while !content.is_empty() {
        if content.peek(Token![try]) {
            let try_token: Token![try] = content.parse()?;

            let mut try_statements = Vec::new();
            parse_code_block(&mut &content, &mut try_statements)?;

            let mut catches = Vec::new();
            let mut catch_all = None;

            let mut go = true;
            while go {
                let content_la = content.fork();
                if let Ok(potential_catch) = content_la.parse::<Ident>() {
                    let pc_str = potential_catch.to_string();
                    match pc_str.as_str() {
                        "catch" => {
                            let mut catch_statements = Vec::new();

                            let _: Ident = content.parse()?;

                            let catch_params;
                            parenthesized!(catch_params in content);

                            let error_type: Ident = catch_params.parse()?;

                            let mut params: Vec<Parameter> = Vec::new();
                            if catch_params.peek(Token![,]) {
                                let _: Token![,] = catch_params.parse()?;

                                let punctuated =
                                    catch_params.parse_terminated(parse_param, Token![,])?;

                                for param in punctuated.into_iter() {
                                    params.push(param);
                                }
                            }

                            parse_code_block(&mut &content, &mut catch_statements)?;

                            catches.push((error_type, params, catch_statements));
                        }
                        "catch_all" => {
                            let _: Ident = content.parse()?;
                            let mut catch_statements = Vec::new();

                            parse_code_block(&mut &content, &mut catch_statements)?;

                            catch_all = Some(catch_statements);
                        }
                        _ => go = false,
                    }
                } else {
                    go = false;
                }
            }

            if !catches.is_empty() || catch_all.is_some() {
                let elem = BodyElement::TryCatch {
                    r#try: try_statements,
                    catches,
                    catch_all,
                };
                body.push(elem);
            } else {
                return Err(syn::Error::new_spanned(
                    try_token,
                    "try requires catch or catch_all",
                ));
            }
        } else if content.peek(Token![if]) {
            parse_if(&mut &content, body)?;
        } else if content.peek(Token![while]) {
            parse_while(&mut &content, body)?;
        // } else if content.peek(Token![for]) {
        //     todo!();
        } else {
            let stmt: Stmt = content.parse()?;
            body.push(BodyElement::Statement(stmt));
        }
    }
    // body = content.call(syn::Block::parse_within)?;

    Ok(())
}

fn parse_if(input: &mut ParseStream, body: &mut Vec<BodyElement>) -> Result<()> {
    let _: Token![if] = input.parse()?;

    let condition: Expr = Expr::parse_without_eager_brace(input)?;

    let mut if_body = Vec::new();
    parse_code_block(input, &mut if_body)?;

    let else_body = if input.peek(Token![else]) {
        let _: Token![else] = input.parse()?;

        let mut else_statements = Vec::new();
        if input.peek(Token![if]) {
            parse_if(input, &mut else_statements)?;
        } else {
            parse_code_block(input, &mut else_statements)?;
        }
        Some(else_statements)
    } else {
        None
    };

    body.push(BodyElement::If {
        condition,
        body: if_body,
        r#else: else_body,
    });

    Ok(())
}

fn parse_while(input: &mut ParseStream, body: &mut Vec<BodyElement>) -> Result<()> {
    let _: Token![while] = input.parse()?;

    let condition: Expr = Expr::parse_without_eager_brace(input)?;

    let mut while_body = Vec::new();
    parse_code_block(input, &mut while_body)?;

    body.push(BodyElement::While {
        condition,
        body: while_body,
    });

    Ok(())
}

fn translate_unary(
    module: &mut WatModule,
    function: &mut WatFunction,
    current_block: &mut InstructionsList,
    expr: &Expr,
    expr_unary: &ExprUnary,
    ty: Option<&WasmType>,
) -> Result<()> {
    match expr_unary.op {
        syn::UnOp::Deref(_) => {
            return Err(syn::Error::new_spanned(
                expr,
                "Dereferencing is not supported",
            ))
        }
        syn::UnOp::Not(_) => {
            translate_expression(
                module,
                function,
                current_block,
                expr_unary.expr.deref(),
                None,
                None,
            )?;

            // let's default to i32 for now
            // TODO: revisit this, maybe it should only default for int literals?
            let ty = match ty {
                Some(ty) => ty,
                None => &WasmType::I32,
            };

            let mut instructions = match ty {
                WasmType::I32 => vec![WatInstruction::I32Eqz],
                WasmType::I64 => vec![WatInstruction::I64Eqz],
                WasmType::F32 => vec![WatInstruction::F32Eqz],
                WasmType::F64 => vec![WatInstruction::F64Eqz],
                WasmType::I8 => vec![WatInstruction::i32_eqz()],
                WasmType::I31Ref => vec![
                    WatInstruction::I31GetS,
                    WatInstruction::I32Eqz,
                    WatInstruction::RefI31,
                ],
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr_unary,
                        format!("not can be used only for numerical types at the moment"),
                    ));
                }
            };
            current_block.append(&mut instructions);
        }
        syn::UnOp::Neg(_) => {
            translate_expression(
                module,
                function,
                current_block,
                expr_unary.expr.deref(),
                None,
                None,
            )?;

            let ty = get_type(module, function, current_block);
            let ty = match ty {
                Some(ty) => ty,
                None => WasmType::I32,
            };

            let mut instructions = match ty {
                // There is no neg instruction for integer types, so for integer types we multiply
                // by -1
                WasmType::I32 => vec![WatInstruction::I32Const(-1), WatInstruction::I32Mul],
                WasmType::I64 => vec![WatInstruction::I64Const(-1), WatInstruction::I64Mul],
                WasmType::F32 => vec![WatInstruction::F32Neg],
                WasmType::F64 => vec![WatInstruction::F64Neg],
                WasmType::I8 => vec![WatInstruction::i32_const(-1), WatInstruction::I32Mul],
                WasmType::I31Ref => vec![
                    WatInstruction::I31GetS,
                    WatInstruction::i32_const(-1),
                    WatInstruction::I32Mul,
                    WatInstruction::RefI31,
                ],
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr_unary,
                        format!("negation can be used only for numeric types, tried on {ty}"),
                    ));
                }
            };
            current_block.append(&mut instructions);
        }
        _ => return Err(syn::Error::new_spanned(expr, "Operation not supported")),
    }

    Ok(())
}

fn translate_binary(
    module: &mut WatModule,
    function: &mut WatFunction,
    current_block: &mut InstructionsList,
    binary: &ExprBinary,
    left_instructions: &mut InstructionsList,
    right_instructions: &mut InstructionsList,
) -> Result<()> {
    let left_ty = get_type(module, function, left_instructions);
    let right_ty = get_type(module, function, right_instructions);

    // TODO: I'm starting to think that I should just bail on anything that is of different types
    // and require explicit conversions. It will be also much easier to document than documenting
    // auto-cast rules
    //
    // negoatiate type to use for the binary operation
    let mut ty = match (&left_ty, &right_ty) {
        (Some(l), Some(r)) if WasmType::compatible_numeric_types(l, r) => {
            WasmType::broader_numeric_type(l, r)
        }
        (Some(l), Some(r)) if l == r => l.clone(),
        (None, None) => {
            return Err(syn::Error::new_spanned(
                binary,
                "Types need to be known for binary operation",
            ))
        }
        (l, r) => {
            return Err(syn::Error::new_spanned(
                binary,
                format!("Types need to match for a binary operation. Left: {l:?}, Right: {r:?}"),
            ))
        }
    };

    // if the negotiated type is I31Ref or I8, we have to convert it to I32 anyway
    if ty == WasmType::I31Ref || ty == WasmType::I8 {
        ty = WasmType::I32;
    }

    // at this point left_ty and right_ty have to be known, so we can safely unwrap()

    let mut left_ty = left_ty.unwrap();
    // i8 is valid only in arrays, getting it out results in I32
    if left_ty == WasmType::I8 {
        left_ty = WasmType::I32;
    }
    if left_ty != ty {
        // the type is different than negoatiated type, so add conversion
        left_instructions.append(&mut left_ty.convert_to_instruction(&ty));
    }

    let mut right_ty = right_ty.unwrap();
    // i8 is valid only in arrays, getting it out results in I32
    if right_ty == WasmType::I8 {
        right_ty = WasmType::I32;
    }
    if right_ty != ty {
        // the type is different than negoatiated type, so add conversion
        right_instructions.append(&mut right_ty.convert_to_instruction(&ty));
    }

    current_block.append(left_instructions);
    current_block.append(right_instructions);

    match binary.op {
        syn::BinOp::Add(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Add,
                WasmType::I64 => WatInstruction::I64Add,
                WasmType::F32 => WatInstruction::F32Add,
                WasmType::F64 => WatInstruction::F64Add,
                WasmType::I8 => WatInstruction::I32Add,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "addition can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Sub(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Sub,
                WasmType::I64 => WatInstruction::I64Sub,
                WasmType::F32 => WatInstruction::F32Sub,
                WasmType::F64 => WatInstruction::F64Sub,
                WasmType::I8 => WatInstruction::I32Sub,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "subtraction can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Mul(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Mul,
                WasmType::I64 => WatInstruction::I64Mul,
                WasmType::F32 => WatInstruction::F32Mul,
                WasmType::F64 => WatInstruction::F64Mul,
                WasmType::I8 => WatInstruction::I32Mul,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "multiplication can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Div(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32DivS,
                WasmType::I64 => WatInstruction::I64DivS,
                WasmType::F32 => WatInstruction::F32Div,
                WasmType::F64 => WatInstruction::F64Div,
                WasmType::I8 => WatInstruction::I32DivS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "division can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Rem(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32RemS,
                WasmType::I64 => WatInstruction::I64RemS,
                WasmType::I8 => WatInstruction::I32RemS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "remainder operation can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::And(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32And,
                WasmType::I64 => WatInstruction::I64And,
                WasmType::I8 => WatInstruction::I32And,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "|| can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Or(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Or,
                WasmType::I64 => WatInstruction::I64Or,
                WasmType::I8 => WatInstruction::I32Or,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "|| can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::BitXor(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Xor,
                WasmType::I64 => WatInstruction::I64Xor,
                WasmType::I8 => WatInstruction::I32Xor,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "bitwise XOR can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::BitAnd(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32And,
                WasmType::I64 => WatInstruction::I64And,
                WasmType::I8 => WatInstruction::I32And,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "bitwise AND can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::BitOr(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Or,
                WasmType::I64 => WatInstruction::I64Or,
                WasmType::I8 => WatInstruction::I32Or,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "bitwise OR can only be performed on integers",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Shl(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Shl,
                WasmType::I64 => WatInstruction::I64Shl,
                WasmType::I8 => WatInstruction::I32Shl,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "only integers can be shifted left",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Shr(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32ShrS,
                WasmType::I64 => WatInstruction::I64ShrS,
                WasmType::I8 => WatInstruction::I32ShrS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "only integers can be shifted right",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Eq(_) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Eq,
                WasmType::I64 => WatInstruction::I64Eq,
                WasmType::F32 => WatInstruction::F32Eq,
                WasmType::F64 => WatInstruction::F64Eq,
                WasmType::I8 => WatInstruction::I32Eq,
                WasmType::I31Ref => WatInstruction::RefEq,
                WasmType::Anyref => WatInstruction::RefEq,
                WasmType::NullRef => WatInstruction::RefEq,
                WasmType::Ref(_, _) => WatInstruction::RefEq,
                WasmType::Array { .. } => WatInstruction::RefEq,
                WasmType::Struct(_) => WatInstruction::RefEq,
                WasmType::Func { .. } => WatInstruction::RefEq,
                WasmType::Tag { .. } => todo!("translate_binary: WasmType::Tag(_) "),
            };

            current_block.push(instruction);
        }
        syn::BinOp::Lt(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32LtS,
                WasmType::I64 => WatInstruction::I64LtS,
                WasmType::F32 => WatInstruction::F32Lt,
                WasmType::F64 => WatInstruction::F64Lt,
                WasmType::I8 => WatInstruction::I32LtS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "less than operation can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Le(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32LeS,
                WasmType::I64 => WatInstruction::I64LeS,
                WasmType::F32 => WatInstruction::F32Le,
                WasmType::F64 => WatInstruction::F64Le,
                WasmType::I8 => WatInstruction::I32LeS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "less then or equal operation can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Ne(_) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Ne,
                WasmType::I64 => WatInstruction::I64Ne,
                WasmType::F32 => WatInstruction::F32Ne,
                WasmType::F64 => WatInstruction::F64Ne,
                WasmType::I8 => WatInstruction::I32Ne,
                WasmType::I31Ref => WatInstruction::RefEq,
                WasmType::Anyref => WatInstruction::RefEq,
                WasmType::NullRef => WatInstruction::RefEq,
                WasmType::Ref(_, _) => WatInstruction::RefEq,
                WasmType::Array { .. } => WatInstruction::RefEq,
                WasmType::Struct(_) => WatInstruction::RefEq,
                WasmType::Func { .. } => WatInstruction::RefEq,
                WasmType::Tag { .. } => todo!("translate_binary: WasmType::Tag(_) "),
            };

            current_block.push(instruction);
        }
        syn::BinOp::Ge(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32GeS,
                WasmType::I64 => WatInstruction::I64GeS,
                WasmType::F32 => WatInstruction::F32Ge,
                WasmType::F64 => WatInstruction::F64Ge,
                WasmType::I8 => WatInstruction::I32GeS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "greater than or equal operation can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Gt(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32GtS,
                WasmType::I64 => WatInstruction::I64GtS,
                WasmType::F32 => WatInstruction::F32Gt,
                WasmType::F64 => WatInstruction::F64Gt,
                WasmType::I8 => WatInstruction::I32GtS,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "greater than or equal operation can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::AddAssign(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Add,
                WasmType::I64 => WatInstruction::I64Add,
                WasmType::F32 => WatInstruction::F32Add,
                WasmType::F64 => WatInstruction::F64Add,
                WasmType::I8 => WatInstruction::I32Add,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "addition can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::SubAssign(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Sub,
                WasmType::I64 => WatInstruction::I64Sub,
                WasmType::F32 => WatInstruction::F32Sub,
                WasmType::F64 => WatInstruction::F64Sub,
                WasmType::I8 => WatInstruction::I32Sub,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "subtraction can only be performed on numeric types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::MulAssign(_) => todo!("translate_binary: syn::BinOp::MulAssign(_) "),
        syn::BinOp::DivAssign(_) => todo!("translate_binary: syn::BinOp::DivAssign(_) "),
        syn::BinOp::RemAssign(_) => todo!("translate_binary: syn::BinOp::RemAssign(_) "),
        syn::BinOp::BitXorAssign(_) => todo!("translate_binary: syn::BinOp::BitXorAssign(_) "),
        syn::BinOp::BitAndAssign(_) => todo!("translate_binary: syn::BinOp::BitAndAssign(_) "),
        syn::BinOp::BitOrAssign(_) => todo!("translate_binary: syn::BinOp::BitOrAssign(_) "),
        syn::BinOp::ShlAssign(_) => todo!("translate_binary: syn::BinOp::ShlAssign(_) "),
        syn::BinOp::ShrAssign(_) => todo!("translate_binary: syn::BinOp::ShrAssign(_) "),
        _ => todo!("translate_binary: _ "),
    }

    Ok(())
}

enum MemoryAccessType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    U128,
    I128,
}

#[derive(Debug, Clone)]
struct MemoryAccessTypeError;

impl FromStr for MemoryAccessType {
    // TODO:implement error handling
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let str = match s {
            "i8" => Self::I8,
            "u8" => Self::U8,
            "i16" => Self::I16,
            "u16" => Self::U16,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "i64" => Self::I64,
            "u64" => Self::U64,
            "i128" => Self::I128,
            "u128" => Self::U128,
            _ => return Err(MemoryAccessTypeError),
        };

        Ok(str)
    }

    type Err = MemoryAccessTypeError;
}

fn get_memory_type(path: &syn::Path) -> Result<Option<MemoryAccessType>> {
    let arguments = &path.segments[0].arguments;
    if let PathArguments::AngleBracketed(AngleBracketedGenericArguments { args, .. }) = arguments {
        if let GenericArgument::Type(Type::Path(path)) = &args[0] {
            let ty: MemoryAccessType =
                path.path.segments[0]
                    .ident
                    .to_string()
                    .parse()
                    .map_err(|_| {
                        syn::Error::new_spanned(path, "Couldn't convert given type to WASM")
                    })?;
            return Ok(Some(ty));
        }
    }

    Ok(None)
}

fn get_type(
    module: &WatModule,
    function: &WatFunction,
    instructions: &[WatInstruction],
) -> Option<WasmType> {
    instructions.last().and_then(|instr| match instr {
        WatInstruction::Nop => todo!("WatInstruction::Local: WatInstruction::Nop "),
        WatInstruction::Local { .. } => {
            todo!("WatInstruction::Local: WatInstruction::Local ")
        }
        WatInstruction::GlobalGet(name) => Some(
            module
                .globals
                .get(name)
                .ok_or(anyhow!("Could not find local {name}"))
                .unwrap()
                .clone()
                .ty,
        ),
        WatInstruction::GlobalSet(_name) => None,
        // TODO: Handle non existent local
        WatInstruction::LocalGet(name) => Some(
            function
                .locals
                .get(name)
                .or(function
                    .params
                    .iter()
                    .find(|(param_name, _)| param_name.as_ref().unwrap_or(&String::new()) == name)
                    .map(|p| &p.1))
                .ok_or(anyhow!("Could not find local {name}"))
                .unwrap()
                .clone(),
        ),
        WatInstruction::LocalSet(_) => todo!("get_type: WatInstruction::LocalSet(_)"),
        WatInstruction::Call(name) => module
            .get_function(name)
            .and_then(|f| f.results.first())
            .cloned(),
        WatInstruction::CallRef(_) => todo!("get_type: WatInstruction::CallRef(_) "),

        WatInstruction::I32Const(_) => Some(WasmType::I32),
        WatInstruction::I64Const(_) => Some(WasmType::I64),
        WatInstruction::F32Const(_) => Some(WasmType::F32),
        WatInstruction::F64Const(_) => Some(WasmType::F64),

        WatInstruction::F64Floor => Some(WasmType::F64),
        WatInstruction::F32Floor => Some(WasmType::F32),

        WatInstruction::F64Trunc => Some(WasmType::F64),
        WatInstruction::F32Trunc => Some(WasmType::F32),

        WatInstruction::F64Inf => Some(WasmType::F64),
        WatInstruction::F64Nan => Some(WasmType::F64),
        WatInstruction::F64NegInf => Some(WasmType::F64),

        WatInstruction::I32Eqz => Some(WasmType::I32),
        WatInstruction::I64Eqz => Some(WasmType::I64),
        WatInstruction::F32Eqz => Some(WasmType::F32),
        WatInstruction::F64Eqz => Some(WasmType::F64),

        WatInstruction::StructNew(_) => todo!("get_type: WatInstruction::StructNew(_) "),
        WatInstruction::StructGet(name, field_name) => {
            if let Some(WasmType::Struct(fields)) = module.get_type_by_name(name) {
                fields
                    .iter()
                    .find(|f| {
                        f.name
                            .as_ref()
                            .is_some_and(|f_name| format!("${f_name}") == *field_name)
                    })
                    .map(|f| f.ty.clone())
            } else {
                None
            }
        }
        WatInstruction::StructSet(_, _) => todo!("get_type: WatInstruction::StructSet(_) "),
        WatInstruction::ArrayNew(_) => todo!("get_type: WatInstruction::ArrayNew(_) "),
        WatInstruction::RefNull(_) => Some(WasmType::NullRef),
        WatInstruction::Ref(_) => todo!("get_type: WatInstruction::Ref(_) "),
        WatInstruction::RefFunc(_) => todo!("get_type: WatInstruction::RefFunc(_) "),
        WatInstruction::Type(_) => todo!("get_type: WatInstruction::Type(_) "),
        WatInstruction::Return => todo!("get_type: WatInstruction::Return "),
        WatInstruction::ReturnCall(_) => todo!("get_type: WatInstruction::ReturnCall(_) "),
        WatInstruction::Block { .. } => todo!("get_type: WatInstruction::Block "),
        WatInstruction::Loop { .. } => todo!("get_type: WatInstruction::Loop "),
        WatInstruction::If { .. } => todo!("get_type: WatInstruction::If "),
        WatInstruction::BrIf(_) => todo!("get_type: WatInstruction::BrIf(_) "),
        WatInstruction::Br(_) => todo!("get_type: WatInstruction::Br(_) "),
        WatInstruction::Empty => todo!("get_type: WatInstruction::Empty "),
        WatInstruction::Log => todo!("get_type: WatInstruction::Log "),
        WatInstruction::Identifier(_) => todo!("get_type: WatInstruction::Identifier(_) "),
        WatInstruction::Drop => todo!("get_type: WatInstruction::Drop "),
        WatInstruction::LocalTee(_) => todo!("get_type: WatInstruction::LocalTee(_) "),
        WatInstruction::RefI31 => Some(WasmType::I31Ref),
        WatInstruction::Throw(_) => todo!("get_type: WatInstruction::Throw(_) "),
        WatInstruction::Try { .. } => todo!("get_type: WatInstruction::Try "),
        WatInstruction::I32Add => Some(WasmType::I32),
        WatInstruction::I64Add => Some(WasmType::I64),
        WatInstruction::F32Add => Some(WasmType::F32),
        WatInstruction::F64Add => Some(WasmType::F64),
        WatInstruction::I32GeS => Some(WasmType::I32),
        WatInstruction::ArrayLen => Some(WasmType::I32),
        WatInstruction::ArrayGet(name) => {
            Some(get_element_type(module, function, name).ok()).flatten()
        }
        WatInstruction::ArrayGetU(_) => Some(WasmType::I32),
        WatInstruction::ArraySet(name) => {
            Some(get_element_type(module, function, name).ok()).flatten()
        }
        WatInstruction::ArrayNewFixed(_, _) => {
            todo!("get_type: WatInstruction::NewFixed")
        }
        WatInstruction::ArrayCopy(_, _) => None,
        WatInstruction::I32Eq => Some(WasmType::I32),
        WatInstruction::I64Eq => Some(WasmType::I32),
        WatInstruction::F32Eq => Some(WasmType::I32),
        WatInstruction::F64Eq => Some(WasmType::I32),
        WatInstruction::I64ExtendI32S => Some(WasmType::I64),
        WatInstruction::I32WrapI64 => Some(WasmType::I32),
        WatInstruction::F64PromoteF32 => Some(WasmType::F64),
        WatInstruction::F32DemoteF64 => Some(WasmType::F32),
        WatInstruction::I32Sub => Some(WasmType::I32),
        WatInstruction::I64Sub => Some(WasmType::I64),
        WatInstruction::F32Sub => Some(WasmType::F32),
        WatInstruction::F64Sub => Some(WasmType::F64),
        WatInstruction::I32Mul => Some(WasmType::I32),
        WatInstruction::I64Mul => Some(WasmType::I64),
        WatInstruction::F32Mul => Some(WasmType::F32),
        WatInstruction::F64Mul => Some(WasmType::F64),
        WatInstruction::I32DivS => Some(WasmType::I32),
        WatInstruction::I64DivS => Some(WasmType::I64),
        WatInstruction::I32DivU => Some(WasmType::I32),
        WatInstruction::I64DivU => Some(WasmType::I64),
        WatInstruction::F32Div => Some(WasmType::F32),
        WatInstruction::F64Div => Some(WasmType::F64),
        WatInstruction::I32RemS => Some(WasmType::I32),
        WatInstruction::I64RemS => Some(WasmType::I64),
        WatInstruction::I32RemU => Some(WasmType::I32),
        WatInstruction::I64RemU => Some(WasmType::I64),
        WatInstruction::I32Ne => Some(WasmType::I32),
        WatInstruction::I64Ne => Some(WasmType::I32),
        WatInstruction::F32Ne => Some(WasmType::I32),
        WatInstruction::F64Ne => Some(WasmType::I32),
        WatInstruction::I32And => Some(WasmType::I32),
        WatInstruction::I64And => Some(WasmType::I32),
        WatInstruction::I32Or => Some(WasmType::I32),
        WatInstruction::I64Or => Some(WasmType::I32),
        WatInstruction::I32Xor => Some(WasmType::I32),
        WatInstruction::I64Xor => Some(WasmType::I32),
        WatInstruction::I32LtS => Some(WasmType::I32),
        WatInstruction::I64LtS => Some(WasmType::I32),
        WatInstruction::I32LtU => Some(WasmType::I32),
        WatInstruction::I64LtU => Some(WasmType::I32),
        WatInstruction::F32Lt => Some(WasmType::I32),
        WatInstruction::F64Lt => Some(WasmType::I32),
        WatInstruction::I32LeS => Some(WasmType::I32),
        WatInstruction::I64LeS => Some(WasmType::I32),
        WatInstruction::I32LeU => Some(WasmType::I32),
        WatInstruction::I64LeU => Some(WasmType::I32),
        WatInstruction::F32Le => Some(WasmType::I32),
        WatInstruction::F64Le => Some(WasmType::I32),
        WatInstruction::I64GeS => Some(WasmType::I32),
        WatInstruction::I32GeU => Some(WasmType::I32),
        WatInstruction::I64GeU => Some(WasmType::I32),
        WatInstruction::F32Ge => Some(WasmType::I32),
        WatInstruction::F64Ge => Some(WasmType::I32),
        WatInstruction::I32GtS => Some(WasmType::I32),
        WatInstruction::I64GtS => Some(WasmType::I32),
        WatInstruction::I32GtU => Some(WasmType::I32),
        WatInstruction::I64GtU => Some(WasmType::I32),
        WatInstruction::F32Gt => Some(WasmType::I32),
        WatInstruction::F64Gt => Some(WasmType::I32),
        WatInstruction::I32Shl => Some(WasmType::I32),
        WatInstruction::I64Shl => Some(WasmType::I64),
        WatInstruction::I32ShrS => Some(WasmType::I32),
        WatInstruction::I64ShrS => Some(WasmType::I64),
        WatInstruction::I32ShrU => Some(WasmType::I32),
        WatInstruction::I64ShrU => Some(WasmType::I64),
        WatInstruction::F32Neg => Some(WasmType::F32),
        WatInstruction::F64Neg => Some(WasmType::F64),
        WatInstruction::I32Store(_) => None,
        WatInstruction::I64Store(_) => None,
        WatInstruction::F32Store(_) => None,
        WatInstruction::F64Store(_) => None,
        WatInstruction::I32Store8(_) => None,
        WatInstruction::I32Store16(_) => None,
        WatInstruction::I64Store8(_) => None,
        WatInstruction::I64Store16(_) => None,
        WatInstruction::I64Store32(_) => None,
        WatInstruction::I32Load(_) => Some(WasmType::I32),
        WatInstruction::I64Load(_) => Some(WasmType::I64),
        WatInstruction::F32Load(_) => Some(WasmType::F32),
        WatInstruction::F64Load(_) => Some(WasmType::F64),
        WatInstruction::I32Load8S(_) => Some(WasmType::I32),
        WatInstruction::I32Load8U(_) => Some(WasmType::I32),
        WatInstruction::I32Load16S(_) => Some(WasmType::I32),
        WatInstruction::I32Load16U(_) => Some(WasmType::I32),
        WatInstruction::I64Load8S(_) => Some(WasmType::I64),
        WatInstruction::I64Load8U(_) => Some(WasmType::I64),
        WatInstruction::I64Load16S(_) => Some(WasmType::I64),
        WatInstruction::I64Load16U(_) => Some(WasmType::I64),
        WatInstruction::I64Load32S(_) => Some(WasmType::I64),
        WatInstruction::I64Load32U(_) => Some(WasmType::I64),
        WatInstruction::F32ConvertI32S => Some(WasmType::F32),
        WatInstruction::F32ConvertI32U => Some(WasmType::F32),
        WatInstruction::F32ConvertI64S => Some(WasmType::F32),
        WatInstruction::F32ConvertI64U => Some(WasmType::F32),
        WatInstruction::F64ConvertI32S => Some(WasmType::F64),
        WatInstruction::F64ConvertI32U => Some(WasmType::F64),
        WatInstruction::F64ConvertI64S => Some(WasmType::F64),
        WatInstruction::F64ConvertI64U => Some(WasmType::F64),
        WatInstruction::I32TruncF32S => Some(WasmType::I32),
        WatInstruction::I32TruncF32U => Some(WasmType::I32),
        WatInstruction::I32TruncF64S => Some(WasmType::I32),
        WatInstruction::I32TruncF64U => Some(WasmType::I32),
        WatInstruction::I64TruncF32S => Some(WasmType::I64),
        WatInstruction::I64TruncF32U => Some(WasmType::I64),
        WatInstruction::I64TruncF64S => Some(WasmType::I64),
        WatInstruction::I64TruncF64U => Some(WasmType::I64),
        WatInstruction::I31GetS => Some(WasmType::I32),
        WatInstruction::I31GetU => Some(WasmType::I32),
        WatInstruction::RefCast(ty) => Some(ty.clone()),
        WatInstruction::RefTest(_) => Some(WasmType::I32),
        WatInstruction::RefEq => Some(WasmType::I32),
        WatInstruction::I32ReinterpretF32 => Some(WasmType::I32),
        WatInstruction::F32ReinterpretI32 => Some(WasmType::F32),
        WatInstruction::I64ReinterpretF64 => Some(WasmType::I64),
        WatInstruction::F64ReinterpretI64 => Some(WasmType::F64),
    })
}

fn extract_name_from_pattern(pattern: &Pat) -> String {
    match pattern {
        syn::Pat::Const(_) => todo!("translate_for_loop: syn::Pat::Const(_) "),
        syn::Pat::Ident(ref ident) => ident.ident.to_string(),
        syn::Pat::Lit(_) => todo!("translate_for_loop: syn::Pat::Lit(_) "),
        syn::Pat::Macro(_) => todo!("translate_for_loop: syn::Pat::Macro(_) "),
        syn::Pat::Or(_) => todo!("translate_for_loop: syn::Pat::Or(_) "),
        syn::Pat::Paren(_) => todo!("translate_for_loop: syn::Pat::Paren(_) "),
        syn::Pat::Path(_) => todo!("translate_for_loop: syn::Pat::Path(_) "),
        syn::Pat::Range(_) => todo!("translate_for_loop: syn::Pat::Range(_) "),
        syn::Pat::Reference(_) => todo!("translate_for_loop: syn::Pat::Reference(_) "),
        syn::Pat::Rest(_) => todo!("translate_for_loop: syn::Pat::Rest(_) "),
        syn::Pat::Slice(_) => todo!("translate_for_loop: syn::Pat::Slice(_) "),
        syn::Pat::Struct(_) => todo!("translate_for_loop: syn::Pat::Struct(_) "),
        syn::Pat::Tuple(_) => todo!("translate_for_loop: syn::Pat::Tuple(_) "),
        syn::Pat::TupleStruct(_) => todo!("translate_for_loop: syn::Pat::TupleStruct(_) "),
        syn::Pat::Type(_) => todo!("translate_for_loop: syn::Pat::Type(_) "),
        syn::Pat::Verbatim(_) => todo!("translate_for_loop: syn::Pat::Verbatim(_) "),
        syn::Pat::Wild(_) => todo!("translate_for_loop: syn::Pat::Wild(_) "),
        _ => todo!("translate_for_loop: _ "),
    }
}

fn get_struct_type(
    module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<String> {
    let ty = get_type_for_a_label(module, function, name)
        .ok_or(anyhow!("Couldn't find a struct with name {name}"))?;

    match ty {
        WasmType::Ref(name, _) => Ok(name.clone()),
        _ => anyhow::bail!("Tried to use type {name} as a struct type"),
    }
}

fn get_array_type(
    module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<String> {
    let ty = get_type_for_a_label(module, function, name)
        .ok_or(anyhow!("Couldn't find array with name {name}"))?;

    match ty {
        WasmType::Ref(name, _) => Ok(name.clone()),
        _ => anyhow::bail!("Tried to use type {name} as an array type"),
    }
}

fn get_var_type(
    module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<WasmType> {
    let ty = get_type_for_a_label(module, function, name)
        .ok_or(anyhow!("Couldn't find local or a global with name {name}"))?;

    Ok(ty.clone())
}

fn get_struct_field_type_by_name(
    module: &WatModule,
    _function: &WatFunction,
    _instructions: &[WatInstruction],
    type_name: &str,
    field_name: &str,
) -> anyhow::Result<WasmType> {
    let ty = module
        .get_type_by_name(type_name)
        .ok_or(anyhow!("Could not find type {type_name}"))?;

    match ty {
        WasmType::Struct(fields) => Ok(fields
            .iter()
            .find(|f| f.name.clone().unwrap() == field_name)
            .unwrap()
            .ty
            .clone()),
        _ => anyhow::bail!("Tried to use type {type_name} as a struct type"),
    }
}

fn get_struct_field_type(
    module: &WatModule,
    _function: &WatFunction,
    _instructions: &[WatInstruction],
    type_name: &str,
    field_index: usize,
) -> anyhow::Result<WasmType> {
    let ty = module
        .get_type_by_name(type_name)
        .ok_or(anyhow!("Could not find type {type_name}"))?;

    match ty {
        WasmType::Struct(fields) => Ok(fields[field_index].ty.clone()),
        _ => anyhow::bail!("Tried to use type {type_name} as a struct type"),
    }
}

fn get_element_type(
    module: &WatModule,
    _function: &WatFunction,
    name: &str,
) -> anyhow::Result<WasmType> {
    let ty = module
        .get_type_by_name(name)
        .ok_or(anyhow!("Could not find type {name}"))?;

    match ty {
        WasmType::Array { ty, .. } => Ok(*ty.clone()),
        _ => anyhow::bail!("Tried to use type {name} as an array type"),
    }
}

fn translate_while_loop(
    module: &mut WatModule,
    function: &mut WatFunction,
    block: &mut InstructionsList,
    condition: &Expr,
    body: &Vec<BodyElement>,
) -> Result<()> {
    let mut instructions = Vec::new();
    translate_expression(
        module,
        function,
        &mut instructions,
        condition,
        None,
        Some(&WasmType::I32),
    )?;

    instructions.push(WatInstruction::i32_eqz());
    instructions.push(WatInstruction::br_if("$block-label"));

    for stmt in body {
        translate_body_element(module, function, &mut instructions, stmt)?;
    }

    instructions.push(WatInstruction::br("$loop-label"));

    // TODO: unique loop labels
    let loop_instr = WatInstruction::r#loop("$loop-label", instructions);
    let block_instr = WatInstruction::block("$block-label", Signature::default(), vec![loop_instr]);

    block.push(block_instr);

    Ok(())
}

fn translate_for_loop(
    module: &mut WatModule,
    function: &mut WatFunction,
    block: &mut InstructionsList,
    for_loop_expr: &ExprForLoop,
) -> Result<()> {
    let var_name = format!("${}", extract_name_from_pattern(for_loop_expr.pat.deref()));
    translate_expression(module, function, block, &for_loop_expr.expr, None, None)?;
    let ty = get_type(module, function, block).unwrap();
    let for_target_local = function.add_local("$for_target", ty.clone());
    let length_local = function.add_local("$length", WasmType::I32);
    let i_local = function.add_local("$i", WasmType::I32);
    block.push(WatInstruction::local_tee(&for_target_local));
    block.push(WatInstruction::ArrayLen);
    block.push(WatInstruction::local_set(&length_local));

    let mut instructions = vec![
        WatInstruction::local_get(&i_local),
        WatInstruction::local_get(&length_local),
        WatInstruction::I32GeS,
        WatInstruction::I32Eqz,
        WatInstruction::br_if("$block-label"),
    ];

    let array_type;
    if let WasmType::Ref(name, _) = ty.clone() {
        array_type = name.clone();
    } else {
        panic!("For loop's target has to be an array type");
    }

    let element_type = get_element_type(module, function, &array_type).unwrap();
    let current_type = if element_type == WasmType::I8 {
        WasmType::I32
    } else {
        element_type.clone()
    };
    function.add_local_exact(&var_name, current_type);

    instructions.push(WatInstruction::local_get(&for_target_local));
    instructions.push(WatInstruction::local_get(&i_local));
    if element_type == WasmType::I8 {
        instructions.push(WatInstruction::array_get_u(&array_type));
    } else {
        instructions.push(WatInstruction::array_get(&array_type));
    }
    instructions.push(WatInstruction::local_set(&var_name));

    //   (array.get $PollablesArray (global.get $pollables) (local.get $i))
    //   (local.set $current)
    //
    for stmt in &for_loop_expr.body.stmts {
        translate_statement(module, function, &mut instructions, stmt)?;
    }

    instructions.push(WatInstruction::local_get(&i_local));
    instructions.push(WatInstruction::i32_const(1));
    instructions.push(WatInstruction::I32Add);
    instructions.push(WatInstruction::local_set(&i_local));

    instructions.push(WatInstruction::br("$loop-label"));

    // // TODO: unique loop labels
    let loop_instr = WatInstruction::r#loop("$loop-label", instructions);
    let block_instr = WatInstruction::block("$block-label", Signature::default(), vec![loop_instr]);

    block.push(block_instr);
    //wat_function.body = instructions.into_iter().map(|i| Box::new(i)).collect();

    Ok(())
}

fn translate_lit(
    _module: &mut WatModule,
    _function: &mut WatFunction,
    current_block: &mut InstructionsList,
    lit: &syn::Lit,
    ty: Option<&WasmType>,
) -> Result<()> {
    let mut instr = match lit {
        syn::Lit::Str(lit_str) => {
            let typeidx = if let Some(WasmType::Ref(typeidx, _)) = ty {
                typeidx
            } else {
                return Err(syn::Error::new_spanned(
                    lit,
                    "string literal may only be assigned to an i8 array",
                ));
            };
            let mut instructions: InstructionsList = lit_str
                .value()
                .as_bytes()
                .iter()
                .map(|b| WatInstruction::I32Const(*b as i32))
                .collect();
            instructions.push(WatInstruction::ArrayNewFixed(
                typeidx.clone(),
                instructions.len() as u16,
            ));
            instructions
        }
        syn::Lit::ByteStr(_) => todo!("translate_lit: syn::Lit::ByteStr(_) "),
        syn::Lit::CStr(_) => todo!("translate_lit: syn::Lit::CStr(_) "),
        syn::Lit::Byte(_) => todo!("translate_lit: syn::Lit::Byte(_) "),
        syn::Lit::Char(lit_char) => {
            vec![WatInstruction::I32Const(lit_char.value() as i32)]
        }
        syn::Lit::Int(lit_int) => {
            // default to i32 if the type is not known
            let ty = ty.unwrap_or(&WasmType::I32);
            match ty {
                WasmType::I32 => vec![WatInstruction::I32Const(lit_int.base10_parse().unwrap())],
                WasmType::I64 => vec![WatInstruction::I64Const(lit_int.base10_parse().unwrap())],
                WasmType::F32 => vec![WatInstruction::F32Const(lit_int.base10_parse().unwrap())],
                WasmType::F64 => vec![WatInstruction::F64Const(lit_int.base10_parse().unwrap())],
                WasmType::I8 => vec![WatInstruction::I32Const(lit_int.base10_parse().unwrap())],
                WasmType::I31Ref => {
                    vec![
                        WatInstruction::I32Const(lit_int.base10_parse().unwrap()),
                        WatInstruction::RefI31,
                    ]
                }
                t => {
                    return Err(syn::Error::new_spanned(
                        lit_int,
                        format!("Not yet implemented, can't translate {t:?} into int"),
                    ))
                }
            }
        }
        syn::Lit::Float(lit_float) => {
            // default to i32 if the type is not known
            let ty = ty.unwrap_or(&WasmType::F32);
            match ty {
                WasmType::F32 => vec![WatInstruction::F32Const(lit_float.base10_parse().unwrap())],
                WasmType::F64 => vec![WatInstruction::F64Const(lit_float.base10_parse().unwrap())],
                t => todo!("translate float lteral: {t:?}"),
            }
        }
        syn::Lit::Bool(_) => todo!("translate_lit: syn::Lit::Bool(_) "),
        syn::Lit::Verbatim(_) => todo!("translate_lit: syn::Lit::Verbatim(_) "),
        _ => todo!("translate_lit: _ "),
    };
    current_block.append(&mut instr);

    Ok(())
}

enum LabelType {
    Global,
    Local,
    Memory,
    Func,
}
fn get_label_type(module: &WatModule, function: &WatFunction, label: &str) -> Option<LabelType> {
    let label = if !label.starts_with("$") {
        format!("${label}")
    } else {
        label.into()
    };

    if module.memories.contains_key(&label) {
        Some(LabelType::Memory)
    } else if module.globals.contains_key(&label) {
        Some(LabelType::Global)
    } else if function.locals.contains_key(&label)
        || function
            .params
            .iter()
            .any(|(name, _)| name.as_ref().unwrap_or(&String::new()) == &label)
    {
        Some(LabelType::Local)
    } else if module
        .functions()
        .iter()
        .any(|f| format!("${}", f.name) == label)
    {
        Some(LabelType::Func)
    } else {
        None
    }
}

fn get_type_for_a_label(
    module: &WatModule,
    function: &WatFunction,
    label: &str,
) -> Option<WasmType> {
    let label = if !label.starts_with("$") {
        format!("${label}")
    } else {
        label.into()
    };

    if module.globals.contains_key(&label) {
        module.globals.get(&label).map(|g| g.ty.clone())
    } else if function.locals.contains_key(&label) {
        function.locals.get(&label).cloned()
    } else if function
        .params
        .iter()
        .any(|(name, _)| name.as_ref().unwrap_or(&String::new()) == &label)
    {
        let ty = function
            .params
            .iter()
            .find(|(name, _)| name.as_ref().unwrap_or(&String::new()) == &label)
            .map(|(_, ty)| ty)
            .cloned();

        ty
    } else {
        None
    }
}

fn get_var_instruction(
    module: &WatModule,
    function: &WatFunction,
    label: &str,
) -> Option<WatInstruction> {
    let label = if !label.starts_with("$") {
        format!("${label}")
    } else {
        label.into()
    };

    match get_label_type(module, function, &label) {
        Some(label_type) => match label_type {
            LabelType::Global => Some(WatInstruction::global_get(label)),
            LabelType::Local => Some(WatInstruction::local_get(label)),
            LabelType::Memory => None,
            LabelType::Func => Some(WatInstruction::ref_func(label)),
        },
        None => None,
    }
}

fn set_var_instruction(
    module: &WatModule,
    function: &WatFunction,
    label: &str,
) -> Option<WatInstruction> {
    let label = if !label.starts_with("$") {
        format!("${label}")
    } else {
        label.into()
    };

    match get_label_type(module, function, &label) {
        Some(label_type) => match label_type {
            LabelType::Global => Some(WatInstruction::global_set(label)),
            LabelType::Local => Some(WatInstruction::local_set(label)),
            LabelType::Memory => None,
            LabelType::Func => None,
        },
        None => None,
    }
}

// TODO: the passing of all of those details is getting ridiculous. I would like to rewrite
// these functions to work on a struct that keeps all the details within a struct, so that
// I don't have to pass everything to each subsequent function call
fn translate_expression(
    module: &mut WatModule,
    function: &mut WatFunction,
    current_block: &mut InstructionsList,
    expr: &Expr,
    _: Option<Semi>,
    ty: Option<&WasmType>,
) -> Result<()> {
    match expr {
        Expr::Array(expr_array) => {
            if let Some(WasmType::Ref(typeidx, _)) = ty {
                // apparently array.new_fixed can fail if the array is over 10k elements
                // https://github.com/dart-lang/sdk/issues/55873
                // not a huge concern for now, but it might be nice to do a check and change
                // strategy based on the elements size
                let length = expr_array.elems.len();
                let elem_type = get_element_type(module, function, typeidx)
                    .map_err(|_| anyhow!("Type needs to be known for a literal"))
                    .unwrap();
                for elem in &expr_array.elems {
                    translate_expression(
                        module,
                        function,
                        current_block,
                        elem,
                        None,
                        Some(&elem_type),
                    )?;
                }
                current_block.push(WatInstruction::ArrayNewFixed(
                    typeidx.to_string(),
                    length as u16,
                ));
            } else {
                return Err(syn::Error::new_spanned(
                    expr_array,
                    format!("Could not get the type for array literal, type we got: {ty:?}"),
                ));
            }
        }
        Expr::Assign(expr_assign) => {
            match *expr_assign.left.clone() {
                Expr::Index(expr_index) => {
                    if let Expr::Path(syn::ExprPath { path, .. }) = expr_index.expr.deref() {
                        let target_name = format!("${}", path.segments[0].ident);
                        let label_type = match get_label_type(module, function, &target_name) {
                            Some(t) => t,
                            None => return Err(syn::Error::new_spanned(
                                path,
                                "${target_name} not found, it's not a global, local nor a memory",
                            )),
                        };

                        match label_type {
                            LabelType::Global => {
                                current_block.push(WatInstruction::global_get(&target_name))
                            }
                            LabelType::Local => {
                                current_block.push(WatInstruction::local_get(&target_name))
                            }
                            LabelType::Memory => {}
                            LabelType::Func => panic!("Can't assign to a function"),
                        }
                        translate_expression(
                            module,
                            function,
                            current_block,
                            expr_index.index.deref(),
                            None,
                            Some(&WasmType::I32),
                        )?;
                        match label_type {
                            LabelType::Global | LabelType::Local => {
                                let array_type =
                                    get_array_type(module, function, &target_name).unwrap();
                                let element_type =
                                    get_element_type(module, function, &array_type).unwrap();
                                translate_expression(
                                    module,
                                    function,
                                    current_block,
                                    expr_assign.right.deref(),
                                    None,
                                    Some(&element_type),
                                )?;
                                current_block.push(WatInstruction::array_set(array_type));
                            }
                            LabelType::Memory => {
                                let memory_type = get_memory_type(path)?;

                                translate_expression(
                                    module,
                                    function,
                                    current_block,
                                    expr_assign.right.deref(),
                                    None,
                                    None,
                                )?;

                                let ty = memory_type.unwrap_or(MemoryAccessType::I32);
                                match ty {
                                    MemoryAccessType::I32 => {
                                        current_block
                                            .push(WatInstruction::I32Store(Some(target_name)));
                                    }
                                    MemoryAccessType::I8 => {
                                        current_block
                                            .push(WatInstruction::I32Store8(Some(target_name)));
                                    }
                                    MemoryAccessType::U16 => {
                                        current_block
                                            .push(WatInstruction::I32Store16(Some(target_name)));
                                    }
                                    t => {
                                        todo!("memory access not implemented");
                                    }
                                }
                            }
                            LabelType::Func => {}
                        }
                    } else {
                        // TODO: this should be tied to the code line
                        panic!("Accessing arrays is only possible by path at the moment");
                    }
                }
                Expr::Path(expr_path) => {
                    let path = format!("${}", expr_path.path.segments[0].ident);
                    let var_type = get_var_type(module, function, &path).unwrap();
                    translate_expression(
                        module,
                        function,
                        current_block,
                        expr_assign.right.deref(),
                        None,
                        Some(&var_type),
                    )?;
                    let instr = set_var_instruction(module, function, &path).ok_or(
                        syn::Error::new_spanned(
                            expr_path,
                            format!("Couldn't find {path} local or global"),
                        ),
                    )?;
                    current_block.push(instr);
                }
                Expr::Field(expr_field) => {
                    translate_expression(
                        module,
                        function,
                        current_block,
                        expr_field.base.as_ref(),
                        None,
                        ty,
                    )?;
                    let struct_type = get_type(module, function, current_block);
                    let type_name = if let Some(WasmType::Ref(name, _)) = struct_type {
                        name.clone()
                    } else {
                        return Err(syn::Error::new_spanned(
                            expr_field,
                            format!("Couldn't determine struct type, found {struct_type:?}"),
                        ));
                    };

                    let (field_name, field_type) = match &expr_field.member {
                        syn::Member::Named(ident) => {
                            let field_type = get_struct_field_type_by_name(
                                module,
                                function,
                                current_block,
                                &type_name,
                                &ident.to_string(),
                            )
                            .unwrap();
                            let field_name = format!("${ident}");
                            (field_name, field_type)
                        }
                        syn::Member::Unnamed(index) => {
                            let field_type = get_struct_field_type(
                                module,
                                function,
                                current_block,
                                &type_name,
                                index.index as usize,
                            )
                            .unwrap();
                            let field_name = format!("{}", index.index);
                            (field_name, field_type)
                        }
                    };
                    translate_expression(
                        module,
                        function,
                        current_block,
                        expr_assign.right.deref(),
                        None,
                        Some(&field_type),
                    )?;

                    current_block.push(WatInstruction::struct_set(&type_name, &field_name));
                }
                e => todo!("assign not implemented for {e:?}"),
            };
        }
        Expr::Async(_) => todo!("translate_expression: Expr::Async(_) "),
        Expr::Await(_) => todo!("translate_expression: Expr::Await(_) "),
        Expr::Binary(binary) => {
            // TODO: handle memories for assign binary operations like x += 1;
            let mut left_instructions = Vec::new();
            let mut right_instructions = Vec::new();
            translate_expression(
                module,
                function,
                &mut left_instructions,
                binary.left.deref(),
                None,
                None,
            )?;
            translate_expression(
                module,
                function,
                &mut right_instructions,
                binary.right.deref(),
                None,
                None,
            )?;

            // TODO: handle casts and/or error handling

            translate_binary(
                module,
                function,
                current_block,
                binary,
                &mut left_instructions,
                &mut right_instructions,
            )?;

            match binary.op {
                syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_) => {
                    // TODO: handle memories here
                    if let Expr::Path(path_expr) = binary.left.deref() {
                        let name = path_expr.path.segments[0].ident.to_string();
                        current_block.push(set_var_instruction(module, function, &name).ok_or(
                            syn::Error::new_spanned(
                                path_expr,
                                format!("Couldn't find {name} local or global"),
                            ),
                        )?);
                    } else {
                        return Err(syn::Error::new_spanned(
                            binary,
                            "left side of the assign statement has to be a path",
                        ));
                    }
                }
                _ => {}
            }
        }
        Expr::Block(expr_block) => {
            for stmt in &expr_block.block.stmts {
                translate_statement(module, function, current_block, stmt)?;
            }
        }
        Expr::Break(_) => current_block.push(WatInstruction::Br("2".to_string())),
        Expr::Call(expr_call) => {
            if let Expr::Path(syn::ExprPath { path, .. }) = expr_call.func.deref() {
                let func_name = path.segments[0].ident.to_string();

                if func_name == "assert" {
                    if expr_call.args.len() != 2 {
                        return Err(syn::Error::new_spanned(
                            &expr_call.args,
                            "assert function requires exactly 2 arguments: condition and message",
                        ));
                    }

                    // Get the condition
                    let mut condition_instructions = Vec::new();
                    translate_expression(
                        module,
                        function,
                        &mut condition_instructions,
                        &expr_call.args[0],
                        None,
                        None,
                    )?;

                    // Get the message
                    if let Expr::Lit(expr_lit) = &expr_call.args[1] {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            let message = lit_str.value();
                            // TODO: I changed Strings handling to UTF16 in order to make it work with
                            // JAWSM as strings in JavaScript are UTF-16 encoded, but in the future I want
                            // to make it configurable
                            let utf16_string: WString<LittleEndian> = WString::from(&message);
                            let (offset, len) = module.add_data(utf16_string.into_bytes());

                            // Generate assertion code
                            current_block.append(&mut condition_instructions);
                            current_block.push(WatInstruction::I32Eqz);

                            // If condition is true (non-zero), log message and throw
                            let then_block = vec![
                                WatInstruction::i32_const(offset as i32),
                                WatInstruction::i32_const(len as i32),
                                WatInstruction::throw("$AssertException".to_string()),
                            ];

                            current_block.push(WatInstruction::r#if(then_block, None));
                        } else {
                            panic!("Second argument of assert must be a string literal");
                        }
                    } else {
                        panic!("Second argument of assert must be a string literal");
                    }
                } else if func_name == "array_copy" {
                    if expr_call.args.len() != 7 {
                        return Err(syn::Error::new_spanned(
                            expr_call.args.clone(),
                            "Expected 7 arguments".to_string(),
                        ));
                    }
                    let typeidx1 = format!("${}", get_expr_path(&expr_call.args[0])?);
                    let typeidx2 = format!("${}", get_expr_path(&expr_call.args[1])?);

                    let types = vec![
                        WasmType::r#ref(typeidx1.clone()),
                        WasmType::I32,
                        WasmType::r#ref(typeidx2.clone()),
                        WasmType::I32,
                        WasmType::I32,
                    ];
                    for (i, arg) in expr_call.args.iter().skip(2).enumerate() {
                        translate_expression(
                            module,
                            function,
                            current_block,
                            arg,
                            None,
                            types.get(i),
                        )?;
                    }
                    current_block.push(WatInstruction::ArrayCopy(typeidx1, typeidx2));
                } else if let Some(label) = get_label_type(module, function, &func_name) {
                    match label {
                        LabelType::Global => {
                            let ty = get_type_for_a_label(module, function, &func_name).ok_or(
                                syn::Error::new_spanned(
                                    expr_call,
                                    format!("Can't find reference {func_name}"),
                                ),
                            )?;
                            // TODO: most of this block of code is the same as for Local. refactor
                            // it
                            let func_type = if let WasmType::Ref(name, _) = ty {
                                name
                            } else {
                                return Err(syn::Error::new_spanned(
                                    expr_call,
                                    "Can't get function type",
                                ));
                            };

                            let params = if let Some(WasmType::Func { name: _, signature }) =
                                module.get_type_by_name(&func_type)
                            {
                                signature.params.clone()
                            } else {
                                return Err(syn::Error::new_spanned(
                                    expr_call,
                                    format!("Couldn't find type for {func_type}"),
                                ));
                            };

                            for (i, arg) in expr_call.args.iter().enumerate() {
                                translate_expression(
                                    module,
                                    function,
                                    current_block,
                                    arg,
                                    None,
                                    params.get(i).map(|p| &p.1),
                                )?;
                            }

                            current_block
                                .push(WatInstruction::global_get(format!("${}", func_name)));
                            current_block.push(WatInstruction::call_ref(func_type));
                        }
                        LabelType::Local => {
                            let ty = get_type_for_a_label(module, function, &func_name).ok_or(
                                syn::Error::new_spanned(
                                    expr_call,
                                    format!("Can't find reference {func_name}"),
                                ),
                            )?;
                            let func_type = if let WasmType::Ref(name, _) = ty {
                                name
                            } else {
                                return Err(syn::Error::new_spanned(
                                    expr_call,
                                    "Can't get function type",
                                ));
                            };

                            let params = if let Some(WasmType::Func { name: _, signature }) =
                                module.get_type_by_name(&func_type)
                            {
                                signature.params.clone()
                            } else {
                                return Err(syn::Error::new_spanned(
                                    expr_call,
                                    format!("Couldn't find type for {func_type}"),
                                ));
                            };

                            for (i, arg) in expr_call.args.iter().enumerate() {
                                translate_expression(
                                    module,
                                    function,
                                    current_block,
                                    arg,
                                    None,
                                    params.get(i).map(|p| &p.1),
                                )?;
                            }

                            current_block
                                .push(WatInstruction::local_get(format!("${}", func_name)));
                            current_block.push(WatInstruction::call_ref(func_type));
                        }
                        LabelType::Memory => todo!("memory access in call"),
                        LabelType::Func => {
                            let params = module.get_function(&func_name).unwrap().params.clone();

                            for (i, arg) in expr_call.args.iter().enumerate() {
                                translate_expression(
                                    module,
                                    function,
                                    current_block,
                                    arg,
                                    None,
                                    params.get(i).map(|p| &p.1),
                                )?;
                            }
                            current_block.push(WatInstruction::call(format!("${}", func_name)));
                        }
                    }
                } else {
                    // panic!("Unknonwn function call {func_name}");
                    current_block.push(WatInstruction::call(format!("${}", func_name)));
                }
            } else {
                panic!("Only calling functions by path is supported at the moment");
            }
        }
        Expr::Cast(expr_cast) => {
            translate_expression(
                module,
                function,
                current_block,
                expr_cast.expr.deref(),
                None,
                ty,
            )?;

            let last_ty_opt = get_type(module, function, current_block);
            if let Some(last_ty) = last_ty_opt {
                let target_type = translate_type(expr_cast.ty.deref())
                    .ok_or(syn::Error::new_spanned(&expr_cast.ty, "Uknown type"))?;

                match last_ty {
                    WasmType::I32 => match target_type {
                        WasmType::I32 => {}
                        WasmType::I64 => current_block.push(WatInstruction::I64ExtendI32S),
                        WasmType::F32 => current_block.push(WatInstruction::F32ConvertI32S),
                        WasmType::F64 => current_block.push(WatInstruction::F64ConvertI32S),
                        WasmType::I31Ref => {
                            current_block.push(WatInstruction::RefI31);
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between i32 and {t}"),
                            ));
                        }
                    },
                    WasmType::I64 => match target_type {
                        WasmType::I32 => current_block.push(WatInstruction::I32WrapI64),
                        WasmType::I64 => {}
                        WasmType::F32 => current_block.push(WatInstruction::F32ConvertI64S),
                        WasmType::F64 => current_block.push(WatInstruction::F64ConvertI64S),
                        WasmType::I31Ref => {
                            current_block.push(WatInstruction::I32WrapI64);
                            current_block.push(WatInstruction::RefI31);
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between i32 and {t}"),
                            ));
                        }
                    },
                    WasmType::F32 => match target_type {
                        WasmType::I32 => current_block.push(WatInstruction::I32TruncF32S),
                        WasmType::I64 => current_block.push(WatInstruction::I64TruncF32S),
                        WasmType::F32 => {}
                        WasmType::F64 => current_block.push(WatInstruction::F64PromoteF32),
                        WasmType::I31Ref => {
                            current_block.push(WatInstruction::I32TruncF32S);
                            current_block.push(WatInstruction::RefI31);
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between f32 and {t}"),
                            ));
                        }
                    },
                    WasmType::F64 => match target_type {
                        WasmType::I32 => current_block.push(WatInstruction::I32TruncF64S),
                        WasmType::I64 => current_block.push(WatInstruction::I64TruncF64S),
                        WasmType::F32 => current_block.push(WatInstruction::F32DemoteF64),
                        WasmType::F64 => {}
                        WasmType::I31Ref => {
                            current_block.push(WatInstruction::I32TruncF64S);
                            current_block.push(WatInstruction::RefI31);
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between f32 and {t}"),
                            ));
                        }
                    },
                    // this should never happen?
                    WasmType::I8 => todo!("convert i8"),
                    WasmType::I31Ref => match target_type {
                        WasmType::I32 => current_block.push(WatInstruction::I31GetS),
                        WasmType::I64 => {
                            current_block.push(WatInstruction::I31GetS);
                            current_block.push(WatInstruction::I64ExtendI32S);
                        }
                        WasmType::F32 => {
                            current_block.push(WatInstruction::I31GetS);
                            current_block.push(WatInstruction::F32ConvertI32S);
                        }
                        WasmType::F64 => {
                            current_block.push(WatInstruction::I31GetS);
                            current_block.push(WatInstruction::F64ConvertI32S);
                        }
                        WasmType::I31Ref => {}
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between i31ref and {t}"),
                            ));
                        }
                    },
                    WasmType::Anyref => match &target_type {
                        WasmType::Ref(_, _) | WasmType::I31Ref => {
                            current_block.push(WatInstruction::ref_cast(target_type));
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between ref anyref and {t}"),
                            ));
                        }
                    },
                    WasmType::Ref(name, nullable) => match &target_type {
                        WasmType::Ref(to_name, to_nullable) => {
                            if &name != to_name {
                                return Err(syn::Error::new_spanned(
                                    &expr_cast.expr,
                                    format!("Can't convert between ref {name} and ref {to_name}"),
                                ));
                            }

                            // we have to convert only if the type names match, but nullable
                            // doesn't
                            if &nullable != to_nullable {
                                current_block.push(WatInstruction::RefCast(target_type));
                            }
                        }
                        t => {
                            return Err(syn::Error::new_spanned(
                                &expr_cast.expr,
                                format!("Can't convert between ref {name} and {t}"),
                            ));
                        }
                    },
                    WasmType::Array { .. } => todo!("as array"),
                    WasmType::Struct(_) => todo!("as struct"),
                    WasmType::Func { .. } => todo!("as func"),
                    WasmType::Tag { .. } => todo!("as tag"),
                    WasmType::NullRef => match &target_type {
                        WasmType::I32 => todo!(),
                        WasmType::I64 => todo!(),
                        WasmType::F32 => todo!(),
                        WasmType::F64 => todo!(),
                        WasmType::I8 => todo!(),
                        WasmType::I31Ref => todo!(),
                        WasmType::Anyref => todo!(),
                        WasmType::NullRef => todo!(),
                        WasmType::Ref(_, nullable) => {
                            if let tarnik_ast::Nullable::False = nullable {
                                return Err(syn::Error::new_spanned(
                                    &expr_cast.expr,
                                    "Can't cast null into a non nullable type",
                                ));
                            } else {
                                // Not sure if there's a better way, but we already issued a null
                                // values, so let's just drop it and issue a better one
                                current_block.push(WatInstruction::Drop);
                                current_block.push(WatInstruction::RefNull(target_type.clone()));
                            }
                        }
                        WasmType::Array { mutable, ty } => todo!(),
                        WasmType::Struct(_) => todo!(),
                        WasmType::Func { name, signature } => todo!(),
                        WasmType::Tag { name, signature } => todo!(),
                    },
                }
            } else {
                return Err(syn::Error::new_spanned(
                    &expr_cast.expr,
                    "Couldn't determine the type to be cast",
                ));
            }
        }
        Expr::Closure(_) => todo!("translate_expression: Expr::Closure(_) "),
        Expr::Const(_) => todo!("translate_expression: Expr::Const(_) "),
        Expr::Continue(_) => current_block.push(WatInstruction::Br("0".to_string())),
        Expr::Field(expr_field) => {
            translate_expression(
                module,
                function,
                current_block,
                expr_field.base.as_ref(),
                None,
                ty,
            )?;
            let struct_type = get_type(module, function, current_block);
            let type_name = if let Some(WasmType::Ref(name, _)) = struct_type {
                name.clone()
            } else {
                return Err(syn::Error::new_spanned(
                    expr_field,
                    format!("Couldn't determine struct type, found {struct_type:?}"),
                ));
            };

            let field_name = match &expr_field.member {
                syn::Member::Named(ident) => format!("${ident}"),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            current_block.push(WatInstruction::struct_get(&type_name, &field_name));
        }
        Expr::ForLoop(for_loop_expr) => {
            translate_for_loop(module, function, current_block, for_loop_expr)?;
        }
        Expr::Group(_) => todo!("translate_expression: Expr::Group(_) "),
        Expr::If(if_expr) => {
            translate_expression(
                module,
                function,
                current_block,
                if_expr.cond.deref(),
                None,
                ty,
            )?;
            let mut then_instructions = Vec::new();

            for stmt in &if_expr.then_branch.stmts {
                translate_statement(module, function, &mut then_instructions, stmt)?;
            }

            let else_instructions = if let Some((_, expr)) = &if_expr.else_branch {
                let mut else_instructions = Vec::new();
                translate_expression(
                    module,
                    function,
                    &mut else_instructions,
                    expr.deref(),
                    None,
                    ty,
                )?;
                Some(else_instructions)
            } else {
                None
            };

            current_block.push(WatInstruction::r#if(then_instructions, else_instructions))
        }
        Expr::Index(expr_index) => {
            if let Expr::Path(syn::ExprPath { path, .. }) = &*expr_index.expr {
                let target_name = format!("${}", path.segments[0].ident);
                let label_type = match get_label_type(module, function, &target_name) {
                    Some(t) => t,
                    None => {
                        return Err(syn::Error::new_spanned(
                            path,
                            "${target_name} not found, it's not a global, local nor a memory",
                        ))
                    }
                };

                match label_type {
                    LabelType::Global => {
                        current_block.push(WatInstruction::global_get(&target_name))
                    }
                    LabelType::Local => current_block.push(WatInstruction::local_get(&target_name)),
                    LabelType::Memory => {}
                    LabelType::Func => panic!("can't index a function reference"),
                }
                translate_expression(
                    module,
                    function,
                    current_block,
                    expr_index.index.deref(),
                    None,
                    Some(&WasmType::I32),
                )?;

                match label_type {
                    LabelType::Global | LabelType::Local => {
                        let array_type = get_array_type(module, function, &target_name).unwrap();

                        let element_type = get_element_type(module, function, &array_type).unwrap();
                        if element_type == WasmType::I8 {
                            current_block.push(WatInstruction::array_get_u(array_type));
                        } else {
                            current_block.push(WatInstruction::array_get(array_type));
                        }
                    }
                    LabelType::Memory => {
                        let memory_type = get_memory_type(path)?;
                        match memory_type {
                            Some(MemoryAccessType::I32) => {
                                current_block.push(WatInstruction::I32Load(target_name.into()));
                            }
                            Some(MemoryAccessType::I8) => {
                                current_block.push(WatInstruction::I32Load8S(target_name.into()));
                            }
                            Some(MemoryAccessType::U16) => {
                                current_block.push(WatInstruction::I32Load16U(target_name.into()));
                            }
                            Some(ty) => todo!("memory access other type"),
                            None => {
                                // default to i32
                                current_block.push(WatInstruction::I32Load(target_name.into()));
                            }
                        }
                    }
                    LabelType::Func => panic!("can't index a function reference"),
                }
            } else {
                return Err(syn::Error::new_spanned(
                    expr,
                    "Arrays can be only accessed by specifying the array name",
                ));
            }
        }
        Expr::Infer(_) => todo!("translate_expression: Expr::Infer(_) "),
        Expr::Let(_) => todo!("translate_expression: Expr::Let(_) "),
        Expr::Lit(expr_lit) => translate_lit(module, function, current_block, &expr_lit.lit, ty)?,
        Expr::Loop(_) => todo!("translate_expression: Expr::Loop(_) "),
        Expr::Macro(expr_macro) => {
            translate_macro(
                module,
                function,
                current_block,
                &expr_macro.mac.path.segments[0].ident,
                expr_macro.mac.tokens.clone(),
                expr_macro.span(),
            )?;
        }
        Expr::Match(_) => todo!("translate_expression: Expr::Match(_) "),
        Expr::MethodCall(_) => todo!("translate_expression: Expr::MethodCall(_) "),
        Expr::Paren(expr_paren) => {
            // TODO: it kinda looks like this implementation is too simple, am I missing something?
            translate_expression(
                module,
                function,
                current_block,
                expr_paren.expr.as_ref(),
                None,
                ty,
            )?;
        }
        Expr::Path(path_expr) => {
            let name = path_expr.path.segments[0].ident.to_string();
            if name == "null" {
                let ty = ty.ok_or(syn::Error::new_spanned(
                    path_expr,
                    "Couldn't get the type for null",
                ))?;
                current_block.push(WatInstruction::ref_null(ty.clone()));
            } else if name == "f64" && path_expr.path.segments.get(1).is_some() {
                if let Some(segment) = path_expr.path.segments.get(1) {
                    let s = segment.ident.to_string();
                    if s == "INFINITY" {
                        current_block.push(WatInstruction::F64Inf);
                    } else if s == "NAN" {
                        current_block.push(WatInstruction::F64Nan);
                    } else if s == "NEG_INFINITY" {
                        current_block.push(WatInstruction::F64NegInf);
                    } else {
                        return Err(syn::Error::new_spanned(
                            path_expr,
                            format!("f64::{s} not recognized"),
                        ));
                    }
                }
            } else {
                current_block.push(get_var_instruction(module, function, &name).ok_or(
                    syn::Error::new_spanned(
                        path_expr,
                        format!("Couldn't find {name} local or global"),
                    ),
                )?);
            }
        }
        Expr::Range(_) => todo!("translate_expression: Expr::Range(_) "),
        Expr::RawAddr(_) => todo!("translate_expression: Expr::RawAddr(_) "),
        Expr::Reference(_) => todo!("translate_expression: Expr::Reference(_) "),
        Expr::Repeat(expr_repeat) => {
            if let Some(WasmType::Ref(typeidx, _)) = ty {
                // apparently array.new_fixed can fail if the array is over 10k elements
                // https://github.com/dart-lang/sdk/issues/55873
                // not a huge concern for now, but it might be nice to do a check and change
                // strategy based on the elements size
                let elem_type = get_element_type(module, function, typeidx)
                    .map_err(|_| anyhow!("Type needs to be known for a literal"))
                    .unwrap();
                // for elem in &expr_array.elems {

                // }
                translate_expression(
                    module,
                    function,
                    current_block,
                    expr_repeat.expr.deref(),
                    None,
                    Some(&elem_type),
                )?;
                translate_expression(
                    module,
                    function,
                    current_block,
                    expr_repeat.len.deref(),
                    None,
                    Some(&WasmType::I32),
                )?;
                current_block.push(WatInstruction::array_new(typeidx.to_string()));
            } else {
                return Err(syn::Error::new_spanned(
                    expr_repeat,
                    format!("Could not get the type for array literal, type we got: {ty:?}"),
                ));
            }
        }
        Expr::Return(ret) => {
            if let Some(expr) = &ret.expr {
                let results = function.results.clone();
                // This assumes we have only one result. For now it's fine as we don't support
                // more, but it might change in the future
                let result_type = results.first();
                translate_expression(module, function, current_block, expr, None, result_type)?;
            }
            current_block.push(WatInstruction::Return);
        }
        Expr::Struct(expr_struct) => {
            let ident = &expr_struct.path.segments[0].ident;
            let type_name = format!("${ident}");

            for (i, field) in expr_struct.fields.iter().cloned().enumerate() {
                let ty =
                    get_struct_field_type(module, function, current_block, &type_name, i).unwrap();
                translate_expression(
                    module,
                    function,
                    current_block,
                    &field.expr,
                    None,
                    Some(&ty),
                )?;
            }
            current_block.push(WatInstruction::struct_new(type_name));
        }
        Expr::Try(_) => todo!("translate_expression: Expr::Try(_) "),
        Expr::TryBlock(_) => todo!("translate_expression: Expr::TryBlock(_) "),
        Expr::Tuple(_) => todo!("translate_expression: Expr::Tuple(_) "),
        Expr::Unary(expr_unary) => {
            translate_unary(module, function, current_block, expr, expr_unary, ty)?;
        }
        Expr::Unsafe(_) => todo!("translate_expression: Expr::Unsafe(_) "),
        Expr::Verbatim(_) => todo!("translate_expression: Expr::Verbatim(_) "),
        Expr::While(_) => todo!("translate_expression: Expr::While(_) "),
        Expr::Yield(_) => todo!("translate_expression: Expr::Yield(_) "),
        _ => todo!("translate_expression: _ "),
    }

    Ok(())
}

fn get_expr_path(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Path(path) => Ok(path.path.segments[0].ident.to_string()),
        _ => return Err(syn::Error::new(expr.span(), "Expected identifier")),
    }
}

#[derive(Debug)]
struct OurWatFunction(WatFunction);
#[derive(Debug)]
struct OurWasmType(WasmType);
#[derive(Debug)]
struct OurSignature(Signature);
#[derive(Debug)]
struct OurWatInstruction(WatInstruction);

impl ToTokens for OurWasmType {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let tokens_str = match &self.0 {
            WasmType::I32 => quote! { tarnik_ast::WasmType::I32 },
            WasmType::I64 => quote! { tarnik_ast::WasmType::I64 },
            WasmType::F32 => quote! { tarnik_ast::WasmType::F32 },
            WasmType::F64 => quote! { tarnik_ast::WasmType::F64 },
            WasmType::I8 => quote! { tarnik_ast::WasmType::I8 },
            WasmType::I31Ref => quote! { tarnik_ast::WasmType::I31Ref },
            WasmType::Anyref => quote! { tarnik_ast::WasmType::Anyref },
            WasmType::NullRef => quote! { tarnik_ast::WasmType::NullRef },
            WasmType::Ref(r, nullable) => {
                let nullable = match nullable {
                    tarnik_ast::Nullable::True => quote! {tarnik_ast::Nullable::True},
                    tarnik_ast::Nullable::False => quote! { tarnik_ast::Nullable::False },
                };
                quote! { tarnik_ast::WasmType::Ref(#r.to_string(), #nullable) }
            }
            WasmType::Array { mutable, ty } => {
                let ty = OurWasmType(*ty.clone());
                quote! { tarnik_ast::WasmType::Array { mutable: #mutable, ty: Box::new(#ty) } }
            }
            WasmType::Struct(s) => {
                let fields = s.iter().map(|field| {
                    let ty = OurWasmType(field.ty.clone());
                    let name = if let Some(name) = &field.name {
                        quote! { Some(#name.to_string()) }
                    } else {
                        quote! { None }
                    };
                    let mutable = field.mutable;

                    quote! { tarnik_ast::StructField { name: #name, ty: #ty, mutable: #mutable } }
                });
                quote! {
                    tarnik_ast::WasmType::Struct(vec![#(#fields),*])
                }
            }
            WasmType::Func { name, signature } => {
                let name = if let Some(name) = &name {
                    quote! { Some(#name.to_string()) }
                } else {
                    quote! { None }
                };
                let result = if let Some(result) = &signature.result {
                    let result = OurWasmType(result.clone());
                    quote! { Some(#result) }
                } else {
                    quote! { None }
                };
                let params = signature.params.clone();
                let params = params.iter().map(|(name, ty)| {
                    let ty = OurWasmType(ty.clone());
                    let name = if let Some(name) = &name {
                        quote! { Some(#name.to_string()) }
                    } else {
                        quote! { None }
                    };
                    quote! { (#name, #ty) }
                });

                quote! {
                    tarnik_ast::WasmType::Func {
                        name: #name,
                        signature: Box::new(tarnik_ast::Signature { params: vec![#(#params),*], result: #result })
                    }
                }
            }
            WasmType::Tag { name, signature } => {
                let result = if let Some(result) = &signature.result {
                    let result = OurWasmType(result.clone());
                    quote! { Some(#result) }
                } else {
                    quote! { None }
                };
                let params = signature.params.clone();
                let params = params.iter().map(|(name, ty)| {
                    let ty = OurWasmType(ty.clone());
                    let name = if let Some(name) = &name {
                        quote! { Some(#name.to_string()) }
                    } else {
                        quote! { None }
                    };
                    quote! { (#name, #ty) }
                });

                quote! {
                    tarnik_ast::WasmType::Tag { name: #name.to_string(), signature: Box::new(tarnik_ast::Signature { params: vec![#(#params),*], result: #result }) }
                }
            }
        };
        tokens.extend(tokens_str);
    }
}

impl ToTokens for OurSignature {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let signature = &self.0;

        let result_q = if let Some(result) = &signature.result {
            let ty = OurWasmType(result.clone());
            quote! { Some(#ty) }
        } else {
            quote! { None }
        };

        let params = signature.params.iter().map(|(maybe_name, ty)| {
            let ty = OurWasmType(ty.clone());
            let name_q = if let Some(name) = maybe_name {
                quote! { Some(#name.to_string()) }
            } else {
                quote! { None }
            };

            quote! { (#name_q, #ty) }
        });

        let tokens_out = quote! {
            tarnik_ast::Signature {
                params: vec![#(#params),*],
                result: #result_q,
            }
        };

        tokens.extend(tokens_out);
    }
}

fn quote_memory_op(name: &str, label: &Option<String>) -> proc_macro2::TokenStream {
    let ident = Ident::new(name, Span::call_site());
    let label_q = if let Some(label) = label {
        quote! { Some(#label.to_string()) }
    } else {
        quote! { None }
    };
    quote! { tarnik_ast::WatInstruction::#ident(#label_q) }
}

impl ToTokens for OurWatInstruction {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        use WatInstruction::*;
        let w = quote! { tarnik_ast::WatInstruction };
        let wat_instruction = &self.0;
        let tokens_str = match &wat_instruction {
            Nop => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Nop ")
            }
            Local { .. } => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Local ")
            }
            GlobalGet(name) => quote! { #w::GlobalGet(#name.to_string()) },
            GlobalSet(name) => quote! { #w::GlobalSet(#name.to_string()) },
            LocalGet(name) => quote! { #w::LocalGet(#name.to_string()) },
            LocalSet(name) => quote! { #w::LocalSet(#name.to_string()) },

            Call(name) => quote! { #w::Call(#name.to_string()) },
            CallRef(name) => quote! { #w::CallRef(#name.to_string()) },

            F64Floor => quote! { #w::F64Floor },
            F32Floor => quote! { #w::F32Floor },

            F64Trunc => quote! { #w::F64Trunc },
            F32Trunc => quote! { #w::F32Trunc },

            F64Inf => quote! { #w::F64Inf },
            F64NegInf => quote! { #w::F64NegInf },
            F64Nan => quote! { #w::F64Nan },

            I32Const(value) => quote! { #w::I32Const(#value) },
            I64Const(value) => quote! { #w::I64Const(#value) },
            F32Const(value) => quote! { #w::F32Const(#value) },
            F64Const(value) => quote! { #w::F64Const(#value) },

            I32Eqz => quote! { #w::I32Eqz },
            I64Eqz => quote! { #w::I64Eqz },
            F32Eqz => quote! { #w::F32Eqz },
            F64Eqz => quote! { #w::F64Eqz },

            StructNew(type_name) => quote! { #w::StructNew(#type_name.to_string()) },
            StructGet(type_name, field_name) => {
                quote! { #w::StructGet(#type_name.to_string(), #field_name.to_string() ) }
            }
            StructSet(type_name, field_name) => {
                quote! { #w::StructSet(#type_name.to_string(), #field_name.to_string() ) }
            }
            ArrayNew(typeidx) => quote! { #w::ArrayNew(#typeidx.to_string()) },
            ArrayCopy(typeidx1, typeidx2) => {
                quote! { #w::ArrayCopy(#typeidx1.to_string(), #typeidx2.to_string()) }
            }
            RefNull(ty) => {
                let ty = OurWasmType(ty.clone());
                quote! { #w::RefNull(#ty) }
            }
            Ref(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Ref(_) ")
            }
            RefFunc(name) => {
                quote! { #w::RefFunc(#name.to_string()) }
            }
            Type(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Type(_) ")
            }
            Return => quote! { #w::Return },
            ReturnCall(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::ReturnCall(_) ")
            }
            Block {
                label,
                signature,
                instructions,
            } => {
                let borrowed = instructions.borrow();
                let instructions = borrowed.iter().map(|i| OurWatInstruction(i.clone()));
                let signature = OurSignature(signature.clone());
                quote! {
                    #w::block(#label, #signature, vec![#(#instructions),*])
                }
            }
            Loop {
                label,
                instructions,
            } => {
                let borrowed = instructions.borrow();
                let instructions = borrowed.iter().map(|i| OurWatInstruction(i.clone()));
                quote! {
                    #w::r#loop(#label, vec![#(#instructions),*])
                }
            }
            If { then, r#else } => {
                let borrowed = then.borrow();
                let then_instructions = borrowed.iter().map(|i| OurWatInstruction(i.clone()));
                let else_code = if let Some(r#else) = r#else {
                    let borrowed = r#else.borrow();
                    let else_instructions = borrowed.iter().map(|i| OurWatInstruction(i.clone()));
                    quote! { Some(vec![#(#else_instructions),*]) }
                } else {
                    quote! { None }
                };

                quote! {
                    #w::r#if(vec![#(#then_instructions),*], #else_code)
                }
            }
            BrIf(label) => quote! { #w::br_if(#label) },
            Br(label) => quote! { #w::br(#label) },
            Empty => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Empty ")
            }
            Log => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Log ")
            }
            Identifier(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Identifier(_) ")
            }
            Drop => {
                quote! { #w::Drop }
            }
            LocalTee(name) => quote! { #w::LocalTee(#name.to_string()) },
            RefI31 => {
                quote! { #w::RefI31 }
            }
            Throw(label) => quote! { #w::Throw(#label.to_string()) },
            Try {
                try_block,
                catches,
                catch_all,
            } => {
                let try_block = try_block.borrow();
                let try_tokens = try_block
                    .iter()
                    .map(|instr| OurWatInstruction(instr.clone()));
                let catches_tokens = catches.iter().map(|(name, instructions)| {
                    let borrowed = instructions.borrow();
                    let instructions = borrowed
                        .iter()
                        .map(|instr| OurWatInstruction(instr.clone()));
                    quote! { (#name.to_string(), vec![#(#instructions),*]) }
                });
                let catch_all_tokens = if let Some(catch_all) = catch_all {
                    let borrowed = catch_all.borrow();
                    let catch_all_instructions =
                        borrowed.iter().map(|i| OurWatInstruction(i.clone()));
                    quote! { Some(vec![#(#catch_all_instructions),*]) }
                } else {
                    quote! { None }
                };

                quote! {
                    #w::r#try(
                        vec![#(#try_tokens),*],
                        vec![#(#catches_tokens),*],
                        #catch_all_tokens,
                    )
                }
            }
            I32Add => quote! { #w::I32Add },
            I64Add => quote! { #w::I64Add },
            F32Add => quote! { #w::F32Add },
            F64Add => quote! { #w::F64Add },
            I32GeS => quote! { #w::I32GeS },
            ArrayLen => quote! { #w::ArrayLen },
            ArrayGet(name) => quote! { #w::ArrayGet(#name.to_string()) },
            ArrayGetU(name) => quote! { #w::ArrayGetU(#name.to_string()) },
            ArraySet(name) => quote! { #w::ArraySet(#name.to_string()) },
            ArrayNewFixed(typeidx, n) => quote! { #w::ArrayNewFixed(#typeidx.to_string(), #n) },
            I32Eq => quote! { #w::I32Eq },
            I64Eq => quote! { #w::I64Eq },
            F32Eq => quote! { #w::F32Eq },
            F64Eq => quote! { #w::F64Eq },
            I64ExtendI32S => quote! { #w::I64ExtendI32S },
            I32WrapI64 => quote! { #w::I32WrapI64 },
            I31GetS => quote! { #w::I31GetS },
            F64PromoteF32 => quote! { #w::F64PromoteF32 },
            F32DemoteF64 => quote! { #w::F32DemoteF64 },
            I32Sub => quote! { #w::I32Sub },
            I64Sub => quote! { #w::I64Sub },
            F32Sub => quote! { #w::F32Sub },
            F64Sub => quote! { #w::F64Sub },
            I32Mul => quote! { #w::I32Mul },
            I64Mul => quote! { #w::I64Mul },
            F32Mul => quote! { #w::F32Mul },
            F64Mul => quote! { #w::F64Mul },
            I32DivS => quote! { #w::I32DivS },
            I64DivS => quote! { #w::I64DivS },
            I32DivU => quote! { #w::I32DivU },
            I64DivU => quote! { #w::I64DivU },
            F32Div => quote! { #w::F32Div },
            F64Div => quote! { #w::F64Div },
            I32RemS => quote! { #w::I32RemS },
            I64RemS => quote! { #w::I64RemS },
            I32RemU => quote! { #w::I32RemU },
            I64RemU => quote! { #w::I64RemU },
            I32And => quote! { #w::I32And },
            I64And => quote! { #w::I64And },
            I32Or => quote! { #w::I32Or },
            I64Or => quote! { #w::I64Or },
            I32Xor => quote! { #w::I32Xor },
            I64Xor => quote! { #w::I64Xor },
            I32Ne => quote! { #w::I32Ne },
            I64Ne => quote! { #w::I64Ne },
            F32Ne => quote! { #w::F32Ne},
            F64Ne => quote! { #w::F64Ne},
            I32LtS => quote! { #w::I32LtS},
            I64LtS => quote! { #w::I64LtS},
            I32LtU => quote! { #w::I32LtU},
            I64LtU => quote! { #w::I64LtU},
            F32Lt => quote! { #w::F32Lt},
            F64Lt => quote! { #w::F64Lt},
            I32LeS => quote! { #w::I32LeS},
            I64LeS => quote! { #w::I64LeS},
            I32LeU => quote! { #w::I32LeU},
            I64LeU => quote! { #w::I64LeU},
            F32Le => quote! { #w::F32Le},
            F64Le => quote! { #w::F64Le},
            I64GeS => quote! { #w::I64GeS},
            I32GeU => quote! { #w::I32GeU},
            I64GeU => quote! { #w::I64GeU},
            F32Ge => quote! { #w::F32Ge},
            F64Ge => quote! { #w::F64Ge},
            I32GtS => quote! { #w::I32GtS},
            I64GtS => quote! { #w::I64GtS},
            I32GtU => quote! { #w::I32GtU},
            I64GtU => quote! { #w::I64GtU},
            F32Gt => quote! { #w::F32Gt},
            F64Gt => quote! { #w::F64Gt},
            I32Shl => quote! { #w::I32Shl},
            I64Shl => quote! { #w::I64Shl},
            I32ShrS => quote! { #w::I32ShrS},
            I64ShrS => quote! { #w::I64ShrS},
            I32ShrU => quote! { #w::I32ShrU},
            I64ShrU => quote! { #w::I64ShrU},
            F32Neg => quote! { #w::F32Neg },
            F64Neg => quote! { #w::F64Neg },
            I32Store(label) => quote_memory_op("I32Store", label),
            I64Store(label) => quote_memory_op("I64Store", label),
            F32Store(label) => quote_memory_op("F32Store", label),
            F64Store(label) => quote_memory_op("F64Store", label),
            I32Store8(label) => quote_memory_op("I32Store8", label),
            I32Store16(label) => quote_memory_op("I32Store16", label),
            I64Store8(label) => quote_memory_op("I64Store8", label),
            I64Store16(label) => quote_memory_op("I64Store16", label),
            I64Store32(label) => quote_memory_op("I64Store32", label),
            I32Load(label) => quote_memory_op("I32Load", label),
            I64Load(label) => quote_memory_op("I64Load", label),
            F32Load(label) => quote_memory_op("F32Load", label),
            F64Load(label) => quote_memory_op("F64Load", label),
            I32Load8S(label) => quote_memory_op("I32Load8S", label),
            I32Load8U(label) => quote_memory_op("I32Load8U", label),
            I32Load16S(label) => quote_memory_op("I32Load16S", label),
            I32Load16U(label) => quote_memory_op("I32Load16U", label),
            I64Load8S(label) => quote_memory_op("I64Load8S", label),
            I64Load8U(label) => quote_memory_op("I64Load8U", label),
            I64Load16S(label) => quote_memory_op("I64Load16S", label),
            I64Load16U(label) => quote_memory_op("I64Load16U", label),
            I64Load32S(label) => quote_memory_op("I64Load32S", label),
            I64Load32U(label) => quote_memory_op("I64Load32U", label),
            F32ConvertI32S => quote! { #w::F32ConvertI32S },
            F32ConvertI32U => quote! { #w::F32ConvertI32U },
            F32ConvertI64S => quote! { #w::F32ConvertI64S },
            F32ConvertI64U => quote! { #w::F32ConvertI64U },
            F64ConvertI32S => quote! { #w::F64ConvertI32S },
            F64ConvertI32U => quote! { #w::F64ConvertI32U },
            F64ConvertI64S => quote! { #w::F64ConvertI64S },
            F64ConvertI64U => quote! { #w::F64ConvertI64U },
            I32TruncF32S => quote! { #w::I32TruncF32S },
            I32TruncF32U => quote! { #w::I32TruncF32U },
            I32TruncF64S => quote! { #w::I32TruncF64S },
            I32TruncF64U => quote! { #w::I32TruncF64U },
            I64TruncF32S => quote! { #w::I64TruncF32S },
            I64TruncF32U => quote! { #w::I64TruncF32U },
            I64TruncF64S => quote! { #w::I64TruncF64S },
            I64TruncF64U => quote! { #w::I64TruncF64U },
            I31GetS => quote! { #w::I31GetS },
            I31GetU => quote! { #w::I31GetU },
            RefCast(ty) => {
                let ty = OurWasmType(ty.clone());
                quote! {
                    #w::RefCast(#ty)
                }
            }
            RefTest(ty) => {
                let ty = OurWasmType(ty.clone());
                quote! {
                    #w::RefTest(#ty)
                }
            }
            RefEq => quote! { #w::RefEq },
            I32ReinterpretF32 => quote! { #w::I32ReinterpretF32 },
            F32ReinterpretI32 => quote! { #w::F32ReinterpretI32 },
            I64ReinterpretF64 => quote! { #w::I64ReinterpretF64 },
            F64ReinterpretI64 => quote! { #w::F64ReinterpretI64 },
        };
        tokens.extend(tokens_str);
    }
}

impl ToTokens for OurWatFunction {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let wat_function = &self.0;
        let name = &wat_function.name;

        let params = wat_function.params.iter().map(|(name, ty)| {
            let ty = OurWasmType(ty.clone());
            quote! {
                function.add_param(#name, &#ty)
            }
        });

        let body = wat_function.body.borrow();
        let instructions = body.iter().map(|i| {
            let instruction = OurWatInstruction(i.clone());
            quote! { function.add_instruction(#instruction) }
        });

        let results = wat_function.results.iter().map(|ty| {
            let ty = OurWasmType(ty.clone());
            quote! {
                function.add_result(#ty)
            }
        });

        let locals = wat_function.locals.iter().map(|(name, ty)| {
            let ty = OurWasmType(ty.clone());
            quote! { function.add_local_exact(#name.to_string(), #ty) }
        });

        tokens.extend(quote! {
            let mut function = tarnik_ast::WatFunction::new(#name);

            #(#params);*;
            #(#results);*;
            #(#locals);*;
            #(#instructions);*;
        });
    }
}

fn translate_type(ty: &Type) -> Option<WasmType> {
    match ty {
        syn::Type::Array(_) => {
            todo!("Stmt::Local(local): syn::Type::Array(_) {ty:#?}")
        }
        syn::Type::BareFn(bare_fn) => Some(WasmType::Func {
            name: None,
            signature: Box::new(Signature {
                params: bare_fn
                    .inputs
                    .iter()
                    .map(|i| (None, translate_type(&i.ty).unwrap()))
                    .collect(),
                result: match &bare_fn.output {
                    syn::ReturnType::Default => None,
                    syn::ReturnType::Type(_, ty) => translate_type(ty.as_ref()),
                },
            }),
        }),
        syn::Type::Group(_) => {
            todo!("Stmt::Local(local): syn::Type::Group(_) ")
        }
        syn::Type::ImplTrait(_) => {
            todo!("Stmt::Local(local): syn::Type::ImplTrait(_) ")
        }
        syn::Type::Infer(_) => {
            todo!("Stmt::Local(local): syn::Type::Infer(_) ")
        }
        syn::Type::Macro(_) => {
            todo!("Stmt::Local(local): syn::Type::Macro(_) ")
        }
        syn::Type::Never(_) => {
            todo!("Stmt::Local(local): syn::Type::Never(_) ")
        }
        syn::Type::Paren(_) => {
            todo!("Stmt::Local(local): syn::Type::Paren(_) ")
        }
        syn::Type::Path(type_path) => {
            let name = type_path.path.segments[0].ident.to_string();
            let ty = match &type_path.path.segments[0].arguments {
                syn::PathArguments::None => None,
                syn::PathArguments::AngleBracketed(arg) => match &arg.args[0] {
                    syn::GenericArgument::Lifetime(_) => {
                        todo!("translate_type: syn::GenericArgument::Lifetime(_) ")
                    }
                    syn::GenericArgument::Type(ty) => translate_type(ty),
                    syn::GenericArgument::Const(_) => {
                        todo!("translate_type, GenericArgument::Const")
                    }
                    syn::GenericArgument::AssocType(_) => {
                        todo!("translate_type: syn::GenericArgument::AssocType(_) ")
                    }
                    syn::GenericArgument::AssocConst(_) => {
                        todo!("translate_type: syn::GenericArgument::AssocConst(_) ")
                    }
                    syn::GenericArgument::Constraint(_) => {
                        todo!("translate_type: syn::GenericArgument::Constraint(_) ")
                    }
                    _ => todo!("translate_type: _ "),
                },
                syn::PathArguments::Parenthesized(_) => {
                    todo!("translate_type: syn::PathArguments::Parenthesized(_) ")
                }
            };

            match (name.as_str(), ty) {
                ("Nullable", Some(WasmType::Ref(ty, _))) => {
                    Some(WasmType::Ref(ty, tarnik_ast::Nullable::True))
                }
                ("Nullable", Some(ty)) => match ty {
                    WasmType::Anyref => Some(WasmType::Anyref),
                    _ => unimplemented!("Only ref types are nullable"),
                },
                (_, Some(_)) => {
                    unimplemented!("Only Nullable is available as a wrapper type")
                }
                (_, None) => Some(name.parse().unwrap()),
            }
        }
        syn::Type::Ptr(_) => {
            todo!("Stmt::Local(local): syn::Type::Ptr(_) ")
        }
        syn::Type::Reference(_) => {
            todo!("Stmt::Local(local): syn::Type::Reference(_) ")
        }
        syn::Type::Slice(type_slice) => {
            if let syn::Type::Path(path) = *type_slice.elem.clone() {
                let ty: WasmType = path.path.segments[0].ident.to_string().parse().unwrap();

                Some(WasmType::Array {
                    mutable: false,
                    ty: Box::new(ty),
                })
            } else {
                unimplemented!("The only possible array definition is with the identifier inside")
            }
        }
        syn::Type::TraitObject(_) => {
            todo!("Stmt::Local(local): syn::Type::TraitObject(_) ")
        }
        syn::Type::Tuple(_) => {
            todo!("Stmt::Local(local): syn::Type::Tuple(_) ")
        }
        syn::Type::Verbatim(_) => {
            todo!("Stmt::Local(local): syn::Type::Verbatim(_) ")
        }
        _ => todo!("Stmt::Local(local): _ "),
    }
}

fn translate_pat_type(
    _function: &mut WatFunction,
    pat_type: &PatType,
) -> (String, Option<WasmType>) {
    if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
        let name = pat_ident.ident.to_string();
        let ty = translate_type(&pat_type.ty);
        (format!("${name}"), ty)
    } else {
        unimplemented!("Only let with identifiers is implemented")
    }
}

fn translate_macro(
    module: &mut WatModule,
    function: &mut WatFunction,
    instructions: &mut InstructionsList,
    ident: &Ident,
    tokens: TokenStream2,
    macro_span: Span,
) -> Result<()> {
    let name = ident.to_string();
    match name.as_ref() {
        "data" => {
            ::syn::parse::Parser::parse2(
                |input: ParseStream<'_>| {
                    let data_string: LitStr = input.parse()?;
                    // TODO: I changed Strings handling to UTF16 in order to make it work with
                    // JAWSM as strings in JavaScript are UTF-16 encoded, but in the future I want
                    // to make it configurable
                    let utf16_string: WString<LittleEndian> = WString::from(&data_string.value());
                    let bytes = utf16_string.into_bytes();
                    let (offset, _) = module.add_data(bytes);
                    instructions.push(WatInstruction::I32Const(offset as i32));

                    Ok(())
                },
                tokens.clone(),
            )?;
        }
        "throw" => {
            ::syn::parse::Parser::parse2(
                |input: ParseStream<'_>| {
                    let error_type: Ident = input.parse()?;

                    if input.peek(Token![,]) {
                        let _: Token![,] = input.parse()?;

                        let punctuated = input.parse_terminated(Expr::parse, Token![,])?;

                        for param_expr in punctuated.into_iter() {
                            translate_expression(
                                module,
                                function,
                                instructions,
                                &param_expr,
                                None,
                                None,
                            )?;
                        }
                    }

                    instructions.push(WatInstruction::throw(format!("${error_type}")));

                    Ok(())
                },
                tokens.clone(),
            )?;
        }
        "len" => {
            let expr: Expr = ::syn::parse::Parser::parse2(Expr::parse, tokens)?;
            if let Expr::Path(expr_path) = expr {
                let name = format!("${}", expr_path.path.segments[0].ident);
                let label_type = get_label_type(module, function, &name);
                match label_type {
                    Some(label_type) => match label_type {
                        LabelType::Global => instructions.push(WatInstruction::GlobalGet(name)),
                        LabelType::Local => instructions.push(WatInstruction::LocalGet(name)),
                        LabelType::Memory => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a memory",
                            ))
                        }
                        LabelType::Func => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a function reference",
                            ))
                        }
                    },
                    None => return Err(syn::Error::new(macro_span, "{name} variable not found")),
                }
                instructions.push(WatInstruction::ArrayLen);
            } else {
                return Err(syn::Error::new(
                    macro_span,
                    "The len!() macro expects an identifier, for example len!(x);",
                ));
            }
        }
        "floor" => {
            let expr: Expr = ::syn::parse::Parser::parse2(Expr::parse, tokens)?;
            if let Expr::Path(expr_path) = expr {
                let name = format!("${}", expr_path.path.segments[0].ident);
                let label_type = get_label_type(module, function, &name);
                match label_type {
                    Some(label_type) => match label_type {
                        LabelType::Global => {
                            instructions.push(WatInstruction::GlobalGet(name.clone()))
                        }
                        LabelType::Local => {
                            instructions.push(WatInstruction::LocalGet(name.clone()))
                        }
                        LabelType::Memory => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a memory",
                            ))
                        }
                        LabelType::Func => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a function reference",
                            ))
                        }
                    },
                    None => return Err(syn::Error::new(macro_span, "{name} variable not found")),
                }

                let ty = get_type_for_a_label(module, function, &name);
                match ty {
                    Some(WasmType::F64) => instructions.push(WatInstruction::F64Floor),
                    Some(WasmType::F32) => instructions.push(WatInstruction::F32Floor),
                    _ => {
                        return Err(syn::Error::new(
                            macro_span,
                            "The trunc operation is only available for f32 and f64",
                        ))
                    }
                }
            } else {
                return Err(syn::Error::new(
                    macro_span,
                    "The floor!() macro expects an identifier, for example floor!(x);",
                ));
            }
        }
        "trunc" => {
            let expr: Expr = ::syn::parse::Parser::parse2(Expr::parse, tokens)?;
            if let Expr::Path(expr_path) = expr {
                let name = format!("${}", expr_path.path.segments[0].ident);
                let label_type = get_label_type(module, function, &name);
                match label_type {
                    Some(label_type) => match label_type {
                        LabelType::Global => {
                            instructions.push(WatInstruction::GlobalGet(name.clone()))
                        }
                        LabelType::Local => {
                            instructions.push(WatInstruction::LocalGet(name.clone()))
                        }
                        LabelType::Memory => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a memory",
                            ))
                        }
                        LabelType::Func => {
                            return Err(syn::Error::new(
                                macro_span,
                                "Can't get a length of a function reference",
                            ))
                        }
                    },
                    None => return Err(syn::Error::new(macro_span, "{name} variable not found")),
                }

                let ty = get_type_for_a_label(module, function, &name);
                match ty {
                    Some(WasmType::F64) => instructions.push(WatInstruction::F64Trunc),
                    Some(WasmType::F32) => instructions.push(WatInstruction::F32Trunc),
                    _ => {
                        return Err(syn::Error::new(
                            macro_span,
                            "The trunc operation is only available for f32 and f64",
                        ))
                    }
                }
            } else {
                return Err(syn::Error::new(
                    macro_span,
                    "The trunc!() macro expects an identifier, for example trunc!(x);",
                ));
            }
        }
        "ref_test" => {
            ::syn::parse::Parser::parse2(
                |mut input: ParseStream<'_>| {
                    let expr: Expr = input.parse()?;

                    let _: Token![,] = input.parse()?;

                    let ty = parse_type(&mut input)?;

                    translate_expression(module, function, instructions, &expr, None, None)?;
                    instructions.push(WatInstruction::ref_test(ty));

                    Ok(())
                },
                tokens.clone(),
            )?;
        }
        _ => {
            return Err(syn::Error::new(
                macro_span,
                format!("Undefined macro {name}"),
            ));
        }
    }

    Ok(())
}

fn translate_statement(
    module: &mut WatModule,
    function: &mut WatFunction,
    instructions: &mut InstructionsList,
    stmt: &Stmt,
) -> Result<()> {
    match stmt {
        Stmt::Local(local) => match &local.pat {
            syn::Pat::Const(_) => todo!("Stmt::Local(local): syn::Pat::Const(_) "),
            syn::Pat::Ident(pat_ident) => {
                let name = pat_ident.ident.to_string();
                if let Some(init) = &local.init {
                    translate_expression(
                        module,
                        function,
                        instructions,
                        &init.expr,
                        Some(Semi::default()),
                        Some(&get_var_type(module, function, &name).unwrap()),
                    )?;
                    instructions.push(set_var_instruction(module, function, &name).ok_or(
                        syn::Error::new_spanned(
                            pat_ident,
                            format!("Couldn't find {name} local or global"),
                        ),
                    )?);
                }
            }
            syn::Pat::Lit(_) => todo!("Stmt::Local(local): syn::Pat::Lit(_) "),
            syn::Pat::Macro(_) => todo!("Stmt::Local(local): syn::Pat::Macro(_) "),
            syn::Pat::Or(_) => todo!("Stmt::Local(local): syn::Pat::Or(_) "),
            syn::Pat::Paren(_) => todo!("Stmt::Local(local): syn::Pat::Paren(_) "),
            syn::Pat::Path(_) => todo!("Stmt::Local(local): syn::Pat::Path(_) "),
            syn::Pat::Range(_) => todo!("Stmt::Local(local): syn::Pat::Range(_) "),
            syn::Pat::Reference(_) => todo!("Stmt::Local(local): syn::Pat::Reference(_) "),
            syn::Pat::Rest(_) => todo!("Stmt::Local(local): syn::Pat::Rest(_) "),
            syn::Pat::Slice(_) => todo!("Stmt::Local(local): syn::Pat::Slice(_) "),
            syn::Pat::Struct(_) => todo!("Stmt::Local(local): syn::Pat::Struct(_) "),
            syn::Pat::Tuple(_) => todo!("Stmt::Local(local): syn::Pat::Tuple(_) "),
            syn::Pat::TupleStruct(_) => {
                todo!("Stmt::Local(local): syn::Pat::TupleStruct(_) ")
            }
            syn::Pat::Type(pat_type) => {
                let (name, ty) = translate_pat_type(function, pat_type);
                let ty = ty
                    .ok_or(anyhow!("Could not translate type in a local statement"))
                    .unwrap(); // type should be known at this point
                function.add_local_exact(&name, ty.clone());

                if let Some(init) = &local.init {
                    translate_expression(
                        module,
                        function,
                        instructions,
                        &init.expr,
                        Some(Semi::default()),
                        Some(&ty),
                    )?;
                    instructions.push(WatInstruction::local_set(name));
                }
            }
            syn::Pat::Verbatim(_) => todo!("Stmt::Local(local): syn::Pat::Verbatim(_) "),
            syn::Pat::Wild(_) => todo!("Stmt::Local(local): syn::Pat::Wild(_) "),
            _ => todo!("Stmt::Local(local): _ "),
        },
        Stmt::Item(_) => todo!("Stmt::Item(_) "),
        Stmt::Expr(expr, semi) => {
            translate_expression(module, function, instructions, expr, *semi, None)?;
            if let Expr::Call(_) = expr {
                // if this is a single function call, it means we're not doing anything with the
                // results, thus we add drop
                let last = instructions.last();
                if let Some(WatInstruction::Call(name)) = last {
                    if let Some(function) = module.get_function(name) {
                        for _ in &function.results {
                            instructions.push(WatInstruction::drop());
                        }
                    }
                }
            }
        }
        Stmt::Macro(stmt_macro) => {
            translate_macro(
                module,
                function,
                instructions,
                &stmt_macro.mac.path.segments[0].ident,
                stmt_macro.mac.tokens.clone(),
                stmt_macro.span(),
            )?;
        }
    }
    Ok(())
}

fn translate_body_element(
    module: &mut WatModule,
    function: &mut WatFunction,
    instructions: &mut InstructionsList,
    elem: &BodyElement,
) -> Result<()> {
    match elem {
        BodyElement::TryCatch {
            r#try,
            catches,
            catch_all,
        } => {
            let mut try_instructions = Vec::new();
            for stmt in r#try {
                translate_body_element(module, function, &mut try_instructions, stmt)?;
            }

            let mut wasm_catches = Vec::new();
            for (ident, params, statements) in catches {
                let mut catch_instructions = Vec::new();

                // first we need to set params
                for param in params.iter().rev() {
                    let name = format!("${}", param.name);
                    function.add_local_exact(name.clone(), param.ty.clone());

                    catch_instructions.push(WatInstruction::local_set(name));
                }

                for stmt in statements {
                    translate_body_element(module, function, &mut catch_instructions, stmt)?;
                }

                wasm_catches.push((format!("${ident}"), catch_instructions));
            }

            let catch_all = if let Some(statements) = &catch_all {
                let mut instructions = Vec::new();
                for stmt in statements {
                    translate_body_element(module, function, &mut instructions, stmt)?;
                }

                Some(instructions)
            } else {
                None
            };

            instructions.push(WatInstruction::r#try(
                try_instructions,
                wasm_catches,
                catch_all,
            ));
        }
        BodyElement::Statement(stmt) => translate_statement(module, function, instructions, stmt)?,
        BodyElement::If {
            condition,
            body,
            r#else,
        } => {
            translate_expression(module, function, instructions, condition, None, None)?;
            let mut then_instructions = Vec::new();

            for stmt in body {
                translate_body_element(module, function, &mut then_instructions, stmt)?;
            }

            let else_instructions = if let Some(body) = r#else {
                let mut else_instructions = Vec::new();
                for stmt in body {
                    translate_body_element(module, function, &mut else_instructions, stmt)?;
                }
                Some(else_instructions)
            } else {
                None
            };

            instructions.push(WatInstruction::r#if(then_instructions, else_instructions))
        }
        BodyElement::While { condition, body } => {
            translate_while_loop(module, function, instructions, condition, body)?;
        }
    }

    Ok(())
}

#[proc_macro]
pub fn wasm(input: TokenStream) -> TokenStream {
    let global_scope = parse_macro_input!(input as GlobalScope);

    let data = global_scope
        .module
        .data
        .clone()
        .into_iter()
        .map(|(offset, data)| {
            let quoted = data.iter().map(|b| quote! { #b });
            quote! {
                module._add_data_raw(#offset, vec![#(#quoted),*]);
            }
        });

    // TODO: this could be moved to WatModule ToTokens
    let types =
        global_scope.module.types.clone().into_iter().map(
            |type_definition| match type_definition {
                TypeDefinition::Rec(types) => {
                    let types_q = types.iter().map(|(name, ty)| {
                        let ty = OurWasmType(ty.clone());
                        quote! {
                            (#name.to_string(), #ty)
                        }
                    });

                    quote! {
                        module.types.push(tarnik_ast::TypeDefinition::Rec(vec![#(#types_q),*]));
                    }
                }
                TypeDefinition::Type(name, ty) => {
                    let ty = OurWasmType(ty);
                    quote! {
                        module.add_type(#name.to_string(), #ty);
                    }
                }
            },
        );

    let tags = global_scope
        .module
        .tags
        .clone()
        .into_iter()
        .map(|(name, type_name)| {
            quote! {
                module.tags.insert(#name.to_string(), #type_name.to_string());
            }
        });

    let exports = global_scope.module.exports.clone().into_iter().map(
        |(export_name, export_type, internal_name)| {
            quote! {
                module.add_export(#export_name, #export_type, #internal_name);
            }
        },
    );

    let imports = global_scope
        .module
        .imports
        .clone()
        .into_iter()
        .map(|(namespace, name, ty)| {
            let ty = OurWasmType(ty);
            quote! {
                module.add_import(#namespace, #name, #ty);
            }
        });

    let globals = global_scope
        .module
        .globals
        .clone()
        .into_iter()
        .map(|(name, global)| {
            let ty = OurWasmType(global.ty);
            let mutable = global.mutable;
            let init = global.init.iter().map(|i| OurWatInstruction(i.clone()));
            quote! {
                module.globals.insert(#name.to_string(), tarnik_ast::Global { name: #name.to_string(), ty: #ty, mutable: #mutable, init: vec![#(#init),*] });
            }
        });

    let memories =
        global_scope
            .module
            .memories
            .clone()
            .into_iter()
            .map(|(label, (size, max_size_opt))| {
                let max_size_q = match max_size_opt {
                    Some(max_size) => quote! { Some(#max_size) },
                    None => quote! { None },
                };
                quote! {
                    module.add_memory(#label, #size, #max_size_q);
                }
            });
    let functions = global_scope
        .module
        .functions()
        .clone()
        .into_iter()
        .map(|f| {
            // Filter out import functions
            if global_scope.module.imports.iter().any(|(_, _, ty)| {
                if let WasmType::Func { name, signature: _ } = ty {
                    format!("${}", f.name) == name.clone().unwrap_or(String::new())
                } else {
                    false
                }
            }) {
                None
            } else {
                let our = OurWatFunction(f.clone());
                Some(quote! {
                    #our

                    module.add_function(function);
                })
            }
        })
        .flatten();

    let output = quote! {
        {
            let mut module = tarnik_ast::WatModule::new();

            #(#imports)*

            #(#data)*

            #(#types)*

            #(#memories)*

            #(#tags)*

            #(#globals)*

            #(#functions)*

            #(#exports)*

            module
        }
    };
    // println!("output:\n{}", output);

    output.into()
}
