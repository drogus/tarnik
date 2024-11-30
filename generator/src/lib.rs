use anyhow::anyhow;
use indexmap::IndexMap;
use proc_macro::TokenStream;
use std::ops::Deref;
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::{self, Semi},
    Attribute, Expr, ExprBinary, ExprForLoop, Lit, Meta, Pat, PatType, Type,
};

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{braced, parenthesized, parse_macro_input, Ident, Result, Stmt, Token};

use tarnik_ast::{StructField, WasmType, WatFunction, WatInstruction, WatModule};

#[derive(Debug, Clone)]
struct Parameter {
    name: String,
    ty: WasmType,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    parameters: Vec<Parameter>,
    #[allow(dead_code)]
    return_type: Option<String>,
    body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
struct GlobalScope {
    types: IndexMap<String, WasmType>,
    functions: Vec<WatFunction>,
    module: WatModule,
}

impl Parse for GlobalScope {
    fn parse(mut input: ParseStream) -> Result<Self> {
        let mut functions = Vec::new();
        let mut types = IndexMap::new();
        let mut export_next: Option<String> = None;

        let mut module = WatModule::new();

        while !input.is_empty() {
            if input.peek(Token![fn]) {
                let function = parse_function(&mut input, &mut functions)?;
                if let Some(export_name) = export_next {
                    module.add_export(export_name, "func", format!("${}", function.name));
                    export_next = None;
                }
            } else if input.peek(Token![type]) {
                let (name, ty) = parse_type_def(&mut input)?;
                types.insert(name.clone(), ty);
                if let Some(export_name) = export_next {
                    module.add_export(export_name, "type", format!("${}", name));
                    export_next = None;
                }
            } else if input.peek(Token![struct]) {
                let (name, ty) = parse_struct(&mut input)?;
                types.insert(format!("${name}"), ty);
                if let Some(export_name) = export_next {
                    module.add_export(export_name, "type", format!("${}", name));
                    export_next = None;
                }
            } else if input.peek(Token![#]) {
                // TODO: extract to a method
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
                        export_next = Some(str.to_string().trim_matches('"').to_string());
                    } else {
                        return Err(syn::Error::new_spanned(
                            &attrs[0].meta,
                            "Export macro accepts only a string literal as an argument",
                        ));
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "Only export macro is supported at the moment",
                    ));
                }
            } else if input.peek(syn::Ident) && input.peek2(Token![!]) {
                let mcro: syn::Macro = input.parse()?;
                let macro_name = &mcro.path.segments[0].ident.to_string();
                if macro_name == "memory" {
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
                        if let Expr::Lit(expr_lit) = name_expr {
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

                    module.add_memory(&name, size, max_size);
                    if let Some(export_name) = export_next {
                        module.add_export(export_name, "memory", format!("${}", name));
                        export_next = None;
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        macro_name,
                        "Only memory macro is supported at the top-level at the moment",
                    ));
                }

                if input.peek(token::Semi) {
                    let _: token::Semi = input.parse()?;
                }
            } else {
                let attr = input.call(Attribute::parse_outer)?;
                println!("attr: {attr:#?}");
                println!("input: {input:#?}");
                todo!("other input type")
            }
        }

        for (name, ty) in types.clone() {
            module.add_type(name.clone(), ty.clone());
        }

        let mut wat_functions = Vec::new();
        for f in functions {
            let mut wat_function = WatFunction::new(&f.name);
            for param in &f.parameters {
                wat_function.add_param(format!("${}", param.name), &param.ty);
            }

            let mut instructions = Vec::new();
            for stmt in &f.body {
                translate_statement(&mut module, &mut wat_function, &mut instructions, stmt)?;
            }
            wat_function.body = instructions.into_iter().map(Box::new).collect();
            wat_functions.push(wat_function);
        }

        Ok(Self {
            functions: wat_functions,
            types,
            module,
        })
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

fn parse_function<'f>(
    input: &mut ParseStream,
    functions: &'f mut Vec<Function>,
) -> Result<&'f Function> {
    let _: Token![fn] = input.parse()?;
    let name: Ident = input.parse()?;

    let content;
    parenthesized!(content in input);

    let mut parameters = Vec::new();
    while !content.is_empty() {
        let name: Ident = content.parse()?;
        let _: Token![:] = content.parse()?;
        let ty: Ident = content.parse()?;

        parameters.push(Parameter {
            name: name.to_string(),
            ty: ty.to_string().parse().unwrap(),
        });

        if !content.is_empty() {
            let _: Token![,] = content.parse()?;
        }
    }

    let return_type = if input.peek(Token![->]) {
        let _: Token![->] = input.parse()?;
        let ty: Ident = input.parse()?;
        Some(ty.to_string())
    } else {
        None
    };

    let content;
    braced!(content in input);
    let body: Vec<Stmt> = content.call(syn::Block::parse_within)?;

    functions.push(Function {
        name: name.to_string(),
        parameters,
        return_type,
        body,
    });

    Ok(functions.last().unwrap())
}

fn translate_binary(
    module: &mut WatModule,
    function: &mut WatFunction,
    current_block: &mut Vec<WatInstruction>,
    binary: &ExprBinary,
    left_instructions: &mut Vec<WatInstruction>,
    right_instructions: &mut Vec<WatInstruction>,
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
        (None, None) => panic!("Types must be known for a binary operation"),
        (l, r) => panic!("Types need to match for a binary operation. Left: {l:?}, Right: {r:?}"),
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
                        "addition can only be performed on number types",
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
                        "subtraction can only be performed on number types",
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
                        "multiplication can only be performed on number types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::Div(op) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Div,
                WasmType::I64 => WatInstruction::I64Div,
                WasmType::F32 => WatInstruction::F32Div,
                WasmType::F64 => WatInstruction::F64Div,
                WasmType::I8 => WatInstruction::I32Div,
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "division can only be performed on number types",
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
        syn::BinOp::And(_) => todo!("translate_binary: syn::BinOp::And(_) "),
        syn::BinOp::Or(_) => todo!("translate_binary: syn::BinOp::Or(_) "),
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
                WasmType::I32 => WatInstruction::I32ShlS,
                WasmType::I64 => WatInstruction::I64ShlS,
                WasmType::I8 => WatInstruction::I32ShlS,
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
                WasmType::I31Ref => todo!("translate_binary: WasmType::I31Ref"),
                WasmType::Anyref => todo!("translate_binary: WasmType::Anyref "),
                WasmType::Ref(_, _) => todo!("translate_binary: WasmType::Ref(_, _) "),
                WasmType::Array { .. } => todo!("translate_binary: WasmType::Array(_) "),
                WasmType::Struct(_) => todo!("translate_binary: WasmType::Struct(_) "),
                WasmType::Func { .. } => todo!("translate_binary: WasmType::Func(_) "),
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
                        "less than operation can only be performed on number types",
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
                        "less then or equal operation can only be performed on number types",
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
                WasmType::I31Ref => todo!("translate_binary: WasmType::I31Ref"),
                WasmType::Anyref => todo!("translate_binary: WasmType::Anyref "),
                WasmType::Ref(_, _) => todo!("translate_binary: WasmType::Ref(_, _) "),
                WasmType::Array { .. } => todo!("translate_binary: WasmType::Array(_) "),
                WasmType::Struct(_) => todo!("translate_binary: WasmType::Struct(_) "),
                WasmType::Func { .. } => todo!("translate_binary: WasmType::Func(_) "),
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
                        "greater than or equal operation can only be performed on number types",
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
                        "greater than or equal operation can only be performed on number types",
                    ));
                }
            };

            current_block.push(instruction);
        }
        syn::BinOp::AddAssign(add_assign) => {}
        syn::BinOp::SubAssign(_) => todo!("translate_binary: syn::BinOp::SubAssign(_) "),
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

fn get_type(
    module: &WatModule,
    function: &WatFunction,
    instructions: &[WatInstruction],
) -> Option<WasmType> {
    instructions
        .last()
        .map(|instr| match instr {
            WatInstruction::Nop => todo!("WatInstruction::Local: WatInstruction::Nop "),
            WatInstruction::Local { .. } => {
                todo!("WatInstruction::Local: WatInstruction::Local ")
            }
            WatInstruction::GlobalGet(_) => {
                todo!("WatInstruction::Local: WatInstruction::GlobalGet(_) ")
            }
            // TODO: Handle non existent local
            WatInstruction::LocalGet(name) => Some(
                function
                    .locals
                    .get(name)
                    .or(function.params.iter().find(|p| &p.0 == name).map(|p| &p.1))
                    .ok_or(anyhow!("Could not find local {name}"))
                    .unwrap()
                    .clone(),
            ),
            WatInstruction::LocalSet(_) => todo!("get_type: WatInstruction::LocalSet(_)"),
            WatInstruction::Call(_) => todo!("get_type: WatInstruction::Call(_) "),

            WatInstruction::I32Const(_) => Some(WasmType::I32),
            WatInstruction::I64Const(_) => Some(WasmType::I64),
            WatInstruction::F32Const(_) => Some(WasmType::F32),
            WatInstruction::F64Const(_) => Some(WasmType::F64),

            WatInstruction::I32Eqz => Some(WasmType::I32),
            WatInstruction::I64Eqz => Some(WasmType::I64),
            WatInstruction::F32Eqz => Some(WasmType::F32),
            WatInstruction::F64Eqz => Some(WasmType::F64),

            WatInstruction::StructNew(_) => todo!("get_type: WatInstruction::StructNew(_) "),
            WatInstruction::StructGet(_, _) => todo!("get_type: WatInstruction::StructGet(_) "),
            WatInstruction::StructSet(_, _) => todo!("get_type: WatInstruction::StructSet(_) "),
            WatInstruction::ArrayNew(_) => todo!("get_type: WatInstruction::ArrayNew(_) "),
            WatInstruction::RefNull(_) => todo!("get_type: WatInstruction::RefNull(_) "),
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
            WatInstruction::Catch(_, _) => todo!("get_type: WatInstruction::Catch(_, "),
            WatInstruction::CatchAll(_) => todo!("get_type: WatInstruction::CatchAll(_) "),
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
            WatInstruction::I32Eq => Some(WasmType::I32),
            WatInstruction::I64Eq => Some(WasmType::I32),
            WatInstruction::F32Eq => Some(WasmType::I32),
            WatInstruction::F64Eq => Some(WasmType::I32),
            WatInstruction::I64ExtendI32S => Some(WasmType::I64),
            WatInstruction::I32WrapI64 => Some(WasmType::I32),
            WatInstruction::I31GetS => Some(WasmType::I32),
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
            WatInstruction::I32Div => Some(WasmType::I32),
            WatInstruction::I64Div => Some(WasmType::I64),
            WatInstruction::F32Div => Some(WasmType::F32),
            WatInstruction::F64Div => Some(WasmType::F64),
            WatInstruction::I32RemS => Some(WasmType::I32),
            WatInstruction::I64RemS => Some(WasmType::I64),
            WatInstruction::I32RemU => Some(WasmType::I32),
            WatInstruction::I64RemU => Some(WasmType::I64),
            WatInstruction::I32Ne => Some(WasmType::I32),
            WatInstruction::I64Ne => Some(WasmType::I64),
            WatInstruction::F32Ne => Some(WasmType::F32),
            WatInstruction::F64Ne => Some(WasmType::F64),
            WatInstruction::I32And => Some(WasmType::I32),
            WatInstruction::I64And => Some(WasmType::I64),
            WatInstruction::I32Or => Some(WasmType::I32),
            WatInstruction::I64Or => Some(WasmType::I64),
            WatInstruction::I32Xor => Some(WasmType::I32),
            WatInstruction::I64Xor => Some(WasmType::I64),
            WatInstruction::I32LtS => Some(WasmType::I32),
            WatInstruction::I64LtS => Some(WasmType::I64),
            WatInstruction::I32LtU => Some(WasmType::I32),
            WatInstruction::I64LtU => Some(WasmType::I64),
            WatInstruction::F32Lt => Some(WasmType::F32),
            WatInstruction::F64Lt => Some(WasmType::F64),
            WatInstruction::I32LeS => Some(WasmType::I32),
            WatInstruction::I64LeS => Some(WasmType::I64),
            WatInstruction::I32LeU => Some(WasmType::I32),
            WatInstruction::I64LeU => Some(WasmType::I64),
            WatInstruction::F32Le => Some(WasmType::F32),
            WatInstruction::F64Le => Some(WasmType::F64),
            WatInstruction::I64GeS => Some(WasmType::I64),
            WatInstruction::I32GeU => Some(WasmType::I32),
            WatInstruction::I64GeU => Some(WasmType::I64),
            WatInstruction::F32Ge => Some(WasmType::F32),
            WatInstruction::F64Ge => Some(WasmType::F64),
            WatInstruction::I32GtS => Some(WasmType::I32),
            WatInstruction::I64GtS => Some(WasmType::I64),
            WatInstruction::I32GtU => Some(WasmType::I32),
            WatInstruction::I64GtU => Some(WasmType::I64),
            WatInstruction::F32Gt => Some(WasmType::F32),
            WatInstruction::F64Gt => Some(WasmType::F64),
            WatInstruction::I32ShlS => Some(WasmType::I32),
            WatInstruction::I64ShlS => Some(WasmType::I64),
            WatInstruction::I32ShlU => Some(WasmType::I32),
            WatInstruction::I64ShlU => Some(WasmType::I64),
            WatInstruction::I32ShrS => Some(WasmType::I32),
            WatInstruction::I64ShrS => Some(WasmType::I64),
            WatInstruction::I32ShrU => Some(WasmType::I32),
            WatInstruction::I64ShrU => Some(WasmType::I64),
        })
        .flatten()
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
    _module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<String> {
    let ty = function
        .locals
        .get(name)
        .ok_or(anyhow!("Couldn't find a struct with name {name}"))?;

    match ty {
        WasmType::Ref(name, _) => Ok(name.clone()),
        _ => anyhow::bail!("Tried to use type {name} as a struct type"),
    }
}

fn get_array_type(
    _module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<String> {
    let ty = function
        .locals
        .get(name)
        .ok_or(anyhow!("Couldn't find array with name {name}"))?;

    match ty {
        WasmType::Ref(name, _) => Ok(name.clone()),
        _ => anyhow::bail!("Tried to use type {name} as an array type"),
    }
}

// TODO: handle globals too
fn get_local_type(
    _module: &WatModule,
    function: &WatFunction,
    name: &str,
) -> anyhow::Result<WasmType> {
    let ty = function
        .locals
        .get(name)
        .ok_or(anyhow!("Could not find local {name}"))?;

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
        .types
        .get(type_name)
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
        .types
        .get(type_name)
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
        .types
        .get(name)
        .ok_or(anyhow!("Could not find type {name}"))?;

    match ty {
        WasmType::Array { ty, .. } => Ok(*ty.clone()),
        _ => anyhow::bail!("Tried to use type {name} as an array type"),
    }
}

fn translate_for_loop(
    module: &mut WatModule,
    function: &mut WatFunction,
    block: &mut Vec<WatInstruction>,
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
        WatInstruction::br_if("$block-label"),
    ];

    let array_type;
    if let WasmType::Ref(name, _) = ty.clone() {
        array_type = name.clone();
    } else {
        panic!("For loop's target has to be an array type");
    }

    let element_type = get_element_type(module, function, &array_type).unwrap();

    function.add_local_exact(&var_name, element_type.clone());

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
    let block_instr = WatInstruction::block("$block-label", vec![loop_instr]);

    block.push(block_instr);
    //wat_function.body = instructions.into_iter().map(|i| Box::new(i)).collect();

    Ok(())
}

fn translate_lit(
    _module: &mut WatModule,
    _function: &mut WatFunction,
    current_block: &mut Vec<WatInstruction>,
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
            let mut instructions: Vec<WatInstruction> = lit_str
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
                WasmType::I31Ref => todo!("i31ref literal"),
                t => todo!("translate int lteral: {t:?}"),
            }
        }
        syn::Lit::Float(_) => todo!("translate_lit: syn::Lit::Float(_) "),
        syn::Lit::Bool(_) => todo!("translate_lit: syn::Lit::Bool(_) "),
        syn::Lit::Verbatim(_) => todo!("translate_lit: syn::Lit::Verbatim(_) "),
        _ => todo!("translate_lit: _ "),
    };
    current_block.append(&mut instr);

    Ok(())
}

// TODO: the passing of all of those details is getting ridiculous. I would like to rewrite
// these functions to work on a struct that keeps all the details within a struct, so that
// I don't have to pass everything to each subsequent function call
fn translate_expression(
    module: &mut WatModule,
    function: &mut WatFunction,
    current_block: &mut Vec<WatInstruction>,
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
                panic!("Could not get the type for array literal, type we got: {ty:?}");
            }
        }
        Expr::Assign(expr_assign) => {
            match *expr_assign.left.clone() {
                Expr::Index(expr_index) => {
                    if let Expr::Path(syn::ExprPath { path, .. }) = expr_index.expr.deref() {
                        let array_name = format!("${}", path.segments[0].ident);
                        current_block.push(WatInstruction::local_get(&array_name));
                        translate_expression(
                            module,
                            function,
                            current_block,
                            expr_index.index.deref(),
                            None,
                            Some(&WasmType::I32),
                        )?;
                        let array_type = get_array_type(module, function, &array_name).unwrap();
                        let element_type = get_element_type(module, function, &array_type).unwrap();
                        translate_expression(
                            module,
                            function,
                            current_block,
                            expr_assign.right.deref(),
                            None,
                            Some(&element_type),
                        )?;
                        current_block.push(WatInstruction::array_set(array_type));
                    } else {
                        // TODO: this should be tied to the code line
                        panic!("Accessing arrays is only possible by path at the moment");
                    }
                }
                Expr::Path(expr_path) => {
                    let path = format!("${}", expr_path.path.segments[0].ident);
                    let local_type = get_local_type(module, function, &path).unwrap();
                    translate_expression(
                        module,
                        function,
                        current_block,
                        expr_assign.right.deref(),
                        None,
                        Some(&local_type),
                    )?;
                    current_block.push(WatInstruction::local_set(path));
                }
                Expr::Field(expr_field) => match &*expr_field.base {
                    Expr::Path(expr_path) => {
                        let ident = &expr_path.path.segments[0].ident;
                        let name = format!("${ident}");
                        let type_name = get_struct_type(module, function, &name).unwrap();
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
                                let field_name = format!("${ident}");
                                (field_name, field_type)
                            }
                        };

                        current_block.push(WatInstruction::local_get(&name));
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
                    _ => panic!("Only path is possible to use with a field access"),
                },
                e => todo!("assign not implemented for {e:?}"),
            };
        }
        Expr::Async(_) => todo!("translate_expression: Expr::Async(_) "),
        Expr::Await(_) => todo!("translate_expression: Expr::Await(_) "),
        Expr::Binary(binary) => {
            println!("BINARY: {binary:#?}");
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
            );
        }
        Expr::Block(expr_block) => {
            for stmt in &expr_block.block.stmts {
                translate_statement(module, function, current_block, stmt)?;
            }
        }
        Expr::Break(_) => todo!("translate_expression: Expr::Break(_) "),
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
                            let (offset, len) = module.add_data(message);

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
                } else {
                    // Regular function call
                    for arg in &expr_call.args {
                        translate_expression(module, function, current_block, arg, None, ty)?;
                    }
                    current_block.push(WatInstruction::call(format!("${}", func_name)));
                }
            } else {
                panic!("Only calling functions by path is supported at the moment");
            }
        }
        Expr::Cast(_) => todo!("translate_expression: Expr::Cast(_) "),
        Expr::Closure(_) => todo!("translate_expression: Expr::Closure(_) "),
        Expr::Const(_) => todo!("translate_expression: Expr::Const(_) "),
        Expr::Continue(_) => todo!("translate_expression: Expr::Continue(_) "),
        Expr::Field(expr_field) => match &*expr_field.base {
            Expr::Path(expr_path) => {
                let ident = &expr_path.path.segments[0].ident;
                let name = format!("${ident}");
                let type_name = get_struct_type(module, function, &name).unwrap();
                let field_name = match &expr_field.member {
                    syn::Member::Named(ident) => format!("${ident}"),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };

                current_block.push(WatInstruction::local_get(&name));
                current_block.push(WatInstruction::struct_get(&type_name, &field_name));
            }
            _ => panic!("Only path is possible to use with a field access"),
        },
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

            let else_instructions = if_expr.else_branch.clone().map(|(_, expr)| {
                let mut else_instructions = Vec::new();
                translate_expression(
                    module,
                    function,
                    &mut else_instructions,
                    expr.deref(),
                    None,
                    ty,
                )
                .unwrap();
                else_instructions
            });

            current_block.push(WatInstruction::r#if(then_instructions, else_instructions))
        }
        Expr::Index(expr_index) => {
            if let Expr::Path(syn::ExprPath { path, .. }) = &*expr_index.expr {
                let array_name = format!("${}", path.segments[0].ident);
                current_block.push(WatInstruction::local_get(&array_name));
                translate_expression(
                    module,
                    function,
                    current_block,
                    expr_index.index.deref(),
                    None,
                    Some(&WasmType::I32),
                )?;
                let array_type = get_array_type(module, function, &array_name).unwrap();

                let element_type = get_element_type(module, function, &array_type).unwrap();
                if element_type == WasmType::I8 {
                    current_block.push(WatInstruction::array_get_u(array_type));
                } else {
                    current_block.push(WatInstruction::array_get(array_type));
                }
            } else {
                // TODO: this should be tied to the code line
                panic!("Only calling functions by path is supported at the moment");
            }
        }
        Expr::Infer(_) => todo!("translate_expression: Expr::Infer(_) "),
        Expr::Let(_) => todo!("translate_expression: Expr::Let(_) "),
        Expr::Lit(expr_lit) => translate_lit(module, function, current_block, &expr_lit.lit, ty)?,
        Expr::Loop(_) => todo!("translate_expression: Expr::Loop(_) "),
        Expr::Macro(_) => todo!("translate_expression: Expr::Macro(_) "),
        Expr::Match(_) => todo!("translate_expression: Expr::Match(_) "),
        Expr::MethodCall(_) => todo!("translate_expression: Expr::MethodCall(_) "),
        Expr::Paren(_) => todo!("translate_expression: Expr::Paren(_) "),
        Expr::Path(path_expr) => {
            let name = path_expr.path.segments[0].ident.to_string();
            current_block.push(WatInstruction::LocalGet(format!("${name}")));
        }
        Expr::Range(_) => todo!("translate_expression: Expr::Range(_) "),
        Expr::RawAddr(_) => todo!("translate_expression: Expr::RawAddr(_) "),
        Expr::Reference(_) => todo!("translate_expression: Expr::Reference(_) "),
        Expr::Repeat(_) => todo!("translate_expression: Expr::Repeat(_) "),
        Expr::Return(ret) => {
            if let Some(expr) = &ret.expr {
                translate_expression(module, function, current_block, expr, None, None)?;
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
        Expr::Unary(_) => todo!("translate_expression: Expr::Unary(_) "),
        Expr::Unsafe(_) => todo!("translate_expression: Expr::Unsafe(_) "),
        Expr::Verbatim(_) => todo!("translate_expression: Expr::Verbatim(_) "),
        Expr::While(_) => todo!("translate_expression: Expr::While(_) "),
        Expr::Yield(_) => todo!("translate_expression: Expr::Yield(_) "),
        _ => todo!("translate_expression: _ "),
    }

    Ok(())
}

#[derive(Debug)]
struct OurWatFunction(WatFunction);
#[derive(Debug)]
struct OurWasmType(WasmType);
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
            WasmType::Func(signature) => {
                let result = if let Some(result) = &signature.result {
                    let result = OurWasmType(result.deref().clone());
                    quote! { Some(#result) }
                } else {
                    quote! { None }
                };
                let params = signature.params.clone();
                let params = params.iter().map(|param| {
                    let param = OurWasmType(param.clone());
                    quote! { #param }
                });

                quote! {
                    tarnik_ast::WasmType::Func(Box::new(tarnik_ast::Signature { params: vec![#(#params),*], result: #result }))
                }
            }
            WasmType::Tag { name, signature } => {
                let result = if let Some(result) = &signature.result {
                    let result = OurWasmType(result.deref().clone());
                    quote! { Some(#result) }
                } else {
                    quote! { None }
                };
                let params = signature.params.clone();
                let params = params.iter().map(|param| {
                    let param = OurWasmType(param.clone());
                    quote! { #param }
                });

                quote! {
                    tarnik_ast::WasmType::Tag { name: #name.to_string(), signature: Box::new(tarnik_ast::Signature { params: vec![#(#params),*], result: #result }) }
                }
            }
        };
        tokens.extend(tokens_str);
    }
}

impl ToTokens for OurWatInstruction {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let tokens_str = match &self.0 {
            WatInstruction::Nop => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Nop ")
            }
            WatInstruction::Local { .. } => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Local ")
            }
            WatInstruction::GlobalGet(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::GlobalGet(_) ")
            }
            WatInstruction::LocalGet(name) => {
                quote! { tarnik_ast::WatInstruction::LocalGet(#name.to_string()) }
            }
            WatInstruction::LocalSet(name) => {
                quote! { tarnik_ast::WatInstruction::LocalSet(#name.to_string()) }
            }
            WatInstruction::Call(name) => {
                quote! { tarnik_ast::WatInstruction::Call(#name.to_string()) }
            }
            WatInstruction::I32Const(value) => {
                quote! { tarnik_ast::WatInstruction::I32Const(#value) }
            }
            WatInstruction::I64Const(value) => {
                quote! { tarnik_ast::WatInstruction::I32Const(#value) }
            }
            WatInstruction::F32Const(value) => {
                quote! { tarnik_ast::WatInstruction::F32Const(#value) }
            }
            WatInstruction::F64Const(value) => {
                quote! { tarnik_ast::WatInstruction::F64Const(#value) }
            }

            WatInstruction::I32Eqz => quote! { tarnik_ast::WatInstruction::I32Eqz },
            WatInstruction::I64Eqz => quote! { tarnik_ast::WatInstruction::I64Eqz },
            WatInstruction::F32Eqz => quote! { tarnik_ast::WatInstruction::F32Eqz },
            WatInstruction::F64Eqz => quote! { tarnik_ast::WatInstruction::F64Eqz },

            WatInstruction::StructNew(type_name) => {
                quote! { tarnik_ast::WatInstruction::StructNew(#type_name.to_string()) }
            }
            WatInstruction::StructGet(type_name, field_name) => {
                quote! { tarnik_ast::WatInstruction::StructGet(#type_name.to_string(), #field_name.to_string() ) }
            }
            WatInstruction::StructSet(type_name, field_name) => {
                quote! { tarnik_ast::WatInstruction::StructSet(#type_name.to_string(), #field_name.to_string() ) }
            }
            WatInstruction::ArrayNew(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::ArrayNew(_) ")
            }
            WatInstruction::RefNull(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::RefNull(_) ")
            }
            WatInstruction::Ref(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Ref(_) ")
            }
            WatInstruction::RefFunc(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::RefFunc(_) ")
            }
            WatInstruction::Type(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Type(_) ")
            }
            WatInstruction::Return => quote! { tarnik_ast::WatInstruction::Return },
            WatInstruction::ReturnCall(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::ReturnCall(_) ")
            }
            WatInstruction::Block {
                label,
                instructions,
            } => {
                let instructions = instructions.iter().map(|i| OurWatInstruction(i.clone()));
                quote! {
                    tarnik_ast::WatInstruction::block(#label, vec![#(#instructions),*])
                }
            }
            WatInstruction::Loop {
                label,
                instructions,
            } => {
                let instructions = instructions.iter().map(|i| OurWatInstruction(i.clone()));
                quote! {
                    tarnik_ast::WatInstruction::r#loop(#label, vec![#(#instructions),*])
                }
            }
            WatInstruction::If { then, r#else } => {
                let then_instructions = then.iter().map(|i| OurWatInstruction(i.clone()));
                let else_code = if let Some(r#else) = r#else {
                    let else_instructions = r#else.iter().map(|i| OurWatInstruction(i.clone()));
                    quote! { Some(vec![#(#else_instructions),*]) }
                } else {
                    quote! { None }
                };

                quote! {
                    tarnik_ast::WatInstruction::If {then: vec![#(#then_instructions),*], r#else: #else_code }
                }
            }
            WatInstruction::BrIf(label) => {
                quote! { tarnik_ast::WatInstruction::br_if(#label) }
            }
            WatInstruction::Br(label) => {
                quote! { tarnik_ast::WatInstruction::br(#label) }
            }
            WatInstruction::Empty => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Empty ")
            }
            WatInstruction::Log => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Log ")
            }
            WatInstruction::Identifier(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Identifier(_) ")
            }
            WatInstruction::Drop => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Drop ")
            }
            WatInstruction::LocalTee(name) => {
                quote! { tarnik_ast::WatInstruction::LocalTee(#name.to_string()) }
            }
            WatInstruction::RefI31 => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::RefI31(_) ")
            }
            WatInstruction::Throw(label) => {
                quote! { tarnik_ast::WatInstruction::Throw(#label.to_string()) }
            }
            WatInstruction::Try { .. } => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Try ")
            }
            WatInstruction::Catch(_, _) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Catch(_, ")
            }
            WatInstruction::CatchAll(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::CatchAll(_) ")
            }
            WatInstruction::I32Add => quote! { tarnik_ast::WatInstruction::I32Add },
            WatInstruction::I64Add => quote! { tarnik_ast::WatInstruction::I64Add },
            WatInstruction::F32Add => quote! { tarnik_ast::WatInstruction::F32Add },
            WatInstruction::F64Add => quote! { tarnik_ast::WatInstruction::F64Add },
            WatInstruction::I32GeS => quote! { tarnik_ast::WatInstruction::I32GeS },
            WatInstruction::ArrayLen => quote! { tarnik_ast::WatInstruction::ArrayLen },
            WatInstruction::ArrayGet(name) => {
                quote! { tarnik_ast::WatInstruction::ArrayGet(#name.to_string()) }
            }
            WatInstruction::ArrayGetU(name) => {
                quote! { tarnik_ast::WatInstruction::ArrayGetU(#name.to_string()) }
            }
            WatInstruction::ArraySet(name) => {
                quote! { tarnik_ast::WatInstruction::ArraySet(#name.to_string()) }
            }
            WatInstruction::ArrayNewFixed(typeidx, n) => {
                quote! { tarnik_ast::WatInstruction::ArrayNewFixed(#typeidx.to_string(), #n) }
            }
            WatInstruction::I32Eq => quote! { tarnik_ast::WatInstruction::I32Eq },
            WatInstruction::I64Eq => quote! { tarnik_ast::WatInstruction::I64Eq },
            WatInstruction::F32Eq => quote! { tarnik_ast::WatInstruction::F32Eq },
            WatInstruction::F64Eq => quote! { tarnik_ast::WatInstruction::F64Eq },
            WatInstruction::I64ExtendI32S => quote! { tarnik_ast::WatInstruction::I64ExtendI32S },
            WatInstruction::I32WrapI64 => quote! { tarnik_ast::WatInstruction::I32WrapI64 },
            WatInstruction::I31GetS => quote! { tarnik_ast::WatInstruction::I31GetS },
            WatInstruction::F64PromoteF32 => quote! { tarnik_ast::WatInstruction::F64PromoteF32 },
            WatInstruction::F32DemoteF64 => quote! { tarnik_ast::WatInstruction::F32DemoteF64 },
            WatInstruction::I32Sub => quote! { tarnik_ast::WatInstruction::I32Sub },
            WatInstruction::I64Sub => quote! { tarnik_ast::WatInstruction::I64Sub },
            WatInstruction::F32Sub => quote! { tarnik_ast::WatInstruction::F32Sub },
            WatInstruction::F64Sub => quote! { tarnik_ast::WatInstruction::F64Sub },
            WatInstruction::I32Mul => quote! { tarnik_ast::WatInstruction::I32Mul },
            WatInstruction::I64Mul => quote! { tarnik_ast::WatInstruction::I64Mul },
            WatInstruction::F32Mul => quote! { tarnik_ast::WatInstruction::F32Mul },
            WatInstruction::F64Mul => quote! { tarnik_ast::WatInstruction::F64Mul },
            WatInstruction::I32Div => quote! { tarnik_ast::WatInstruction::I32Div },
            WatInstruction::I64Div => quote! { tarnik_ast::WatInstruction::I64Div },
            WatInstruction::F32Div => quote! { tarnik_ast::WatInstruction::F32Div },
            WatInstruction::F64Div => quote! { tarnik_ast::WatInstruction::F64Div },
            WatInstruction::I32RemS => quote! { tarnik_ast::WatInstruction::I32RemS },
            WatInstruction::I64RemS => quote! { tarnik_ast::WatInstruction::I64RemS },
            WatInstruction::I32RemU => quote! { tarnik_ast::WatInstruction::I32RemU },
            WatInstruction::I64RemU => quote! { tarnik_ast::WatInstruction::I64RemU },
            WatInstruction::I32And => quote! { tarnik_ast::WatInstruction::I32And },
            WatInstruction::I64And => quote! { tarnik_ast::WatInstruction::I64And },
            WatInstruction::I32Or => quote! { tarnik_ast::WatInstruction::I32Or },
            WatInstruction::I64Or => quote! { tarnik_ast::WatInstruction::I64Or },
            WatInstruction::I32Xor => quote! { tarnik_ast::WatInstruction::I32Xor },
            WatInstruction::I64Xor => quote! { tarnik_ast::WatInstruction::I64Xor },
            WatInstruction::I32Ne => quote! { tarnik_ast::WatInstruction::I32Ne },
            WatInstruction::I64Ne => quote! { tarnik_ast::WatInstruction::I64Ne },
            WatInstruction::F32Ne => quote! { tarnik_ast::WatInstruction::F32Ne},
            WatInstruction::F64Ne => quote! { tarnik_ast::WatInstruction::F64Ne},
            WatInstruction::I32LtS => quote! { tarnik_ast::WatInstruction::I32LtS},
            WatInstruction::I64LtS => quote! { tarnik_ast::WatInstruction::I64LtS},
            WatInstruction::I32LtU => quote! { tarnik_ast::WatInstruction::I32LtU},
            WatInstruction::I64LtU => quote! { tarnik_ast::WatInstruction::I64LtU},
            WatInstruction::F32Lt => quote! { tarnik_ast::WatInstruction::F32Lt},
            WatInstruction::F64Lt => quote! { tarnik_ast::WatInstruction::F64Lt},
            WatInstruction::I32LeS => quote! { tarnik_ast::WatInstruction::I32LeS},
            WatInstruction::I64LeS => quote! { tarnik_ast::WatInstruction::I64LeS},
            WatInstruction::I32LeU => quote! { tarnik_ast::WatInstruction::I32LeU},
            WatInstruction::I64LeU => quote! { tarnik_ast::WatInstruction::I64LeU},
            WatInstruction::F32Le => quote! { tarnik_ast::WatInstruction::F32Le},
            WatInstruction::F64Le => quote! { tarnik_ast::WatInstruction::F64Le},
            WatInstruction::I64GeS => quote! { tarnik_ast::WatInstruction::I64GeS},
            WatInstruction::I32GeU => quote! { tarnik_ast::WatInstruction::I32GeU},
            WatInstruction::I64GeU => quote! { tarnik_ast::WatInstruction::I64GeU},
            WatInstruction::F32Ge => quote! { tarnik_ast::WatInstruction::F32Ge},
            WatInstruction::F64Ge => quote! { tarnik_ast::WatInstruction::F64Ge},
            WatInstruction::I32GtS => quote! { tarnik_ast::WatInstruction::I32GtS},
            WatInstruction::I64GtS => quote! { tarnik_ast::WatInstruction::I64GtS},
            WatInstruction::I32GtU => quote! { tarnik_ast::WatInstruction::I32GtU},
            WatInstruction::I64GtU => quote! { tarnik_ast::WatInstruction::I64GtU},
            WatInstruction::F32Gt => quote! { tarnik_ast::WatInstruction::F32Gt},
            WatInstruction::F64Gt => quote! { tarnik_ast::WatInstruction::F64Gt},
            WatInstruction::I32ShlS => quote! { tarnik_ast::WatInstruction::I32ShlS},
            WatInstruction::I64ShlS => quote! { tarnik_ast::WatInstruction::I64ShlS},
            WatInstruction::I32ShlU => quote! { tarnik_ast::WatInstruction::I32ShlU},
            WatInstruction::I64ShlU => quote! { tarnik_ast::WatInstruction::I64ShlU},
            WatInstruction::I32ShrS => quote! { tarnik_ast::WatInstruction::I32ShrS},
            WatInstruction::I64ShrS => quote! { tarnik_ast::WatInstruction::I64ShrS},
            WatInstruction::I32ShrU => quote! { tarnik_ast::WatInstruction::I32ShrU},
            WatInstruction::I64ShrU => quote! { tarnik_ast::WatInstruction::I64ShrU},
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

        let instructions = wat_function.body.iter().map(|i| {
            let instruction = OurWatInstruction(*i.clone());
            quote! { function.add_instruction(#instruction) }
        });

        let locals = wat_function.locals.iter().map(|(name, ty)| {
            let ty = OurWasmType(ty.clone());
            quote! { function.add_local_exact(#name.to_string(), #ty) }
        });

        tokens.extend(quote! {
            let mut function = tarnik_ast::WatFunction::new(#name);

            #(#params);*;
            #(#locals);*;
            #(#instructions);*;
        });
    }
}

fn translate_type(ty: &Type) -> Option<WasmType> {
    match ty {
        syn::Type::Array(_) => {
            todo!("Stmt::Local(local): syn::Type::Array(_) ")
        }
        syn::Type::BareFn(_) => {
            todo!("Stmt::Local(local): syn::Type::BareFn(_) ")
        }
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
                ("Nullable", Some(_)) => {
                    unimplemented!("Only ref types are nullable")
                }
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

fn translate_statement(
    module: &mut WatModule,
    function: &mut WatFunction,
    instructions: &mut Vec<WatInstruction>,
    stmt: &Stmt,
) -> Result<()> {
    match stmt {
        Stmt::Local(local) => match &local.pat {
            syn::Pat::Const(_) => todo!("Stmt::Local(local): syn::Pat::Const(_) "),
            syn::Pat::Ident(_) => todo!("Stmt::Local(local): syn::Pat::Ident(_) "),
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
        }
        Stmt::Macro(_) => todo!("Stmt::Macro(_) "),
    }

    Ok(())
}

#[proc_macro]
pub fn wasm(input: TokenStream) -> TokenStream {
    let global_scope = parse_macro_input!(input as GlobalScope);

    let data = global_scope.module.data.into_iter().map(|(offset, data)| {
        quote! {
            module.data.push((#offset, #data.to_string()));
        }
    });

    // TODO: this could be moved to WatModule ToTokens
    let types = global_scope.types.into_iter().map(|(name, ty)| {
        let ty = OurWasmType(ty);
        quote! {
            module.add_type(#name.to_string(), #ty);
        }
    });

    let tags = global_scope
        .module
        .tags
        .into_iter()
        .map(|(name, type_name)| {
            quote! {
                module.tags.insert(#name.to_string(), #type_name.to_string());
            }
        });

    let tags =
        global_scope
            .module
            .exports
            .into_iter()
            .map(|(export_name, export_type, internal_name)| {
                quote! {
                    module.add_export(#export_name, #export_type, #internal_name);
                }
            });

    let functions = global_scope.functions.into_iter().map(|f| {
        let our = OurWatFunction(f.clone());
        quote! {
            #our

            module.add_function(function);
        }
    });

    let output = quote! {
        {
            let mut module = tarnik_ast::WatModule::new();

            #(#data)*

            #(#types)*

            #(#tags)*

            #(#functions)*

            module
        }
    };
    // println!("output:\n{}", output);

    output.into()
}
