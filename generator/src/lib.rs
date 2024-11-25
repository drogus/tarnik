use anyhow::{anyhow, bail};
use indexmap::IndexMap;
use proc_macro::TokenStream;
use std::{collections::HashMap, ops::Deref};
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    token::{self, Semi},
    Expr, ExprBinary, ExprForLoop, Pat, PatPath, PatType, Type,
};

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{braced, parenthesized, parse_macro_input, Ident, Result, Stmt, Token};

use wazap_ast::{self as ast, StructField, WasmType, WatFunction, WatInstruction, WatModule};

/// Accumulates multiple errors into a result.
/// Only use this for recoverable errors, i.e. non-parse errors. Fatal errors should early exit to
/// avoid further complications.
macro_rules! extend_errors {
    ($errors: ident, $e: expr) => {
        match $errors {
            Ok(_) => $errors = Err($e),
            Err(ref mut errors) => errors.extend($e),
        }
    };
}

#[derive(Debug, Clone)]
struct Parameter {
    name: String,
    ty: WasmType,
}

// impl ToTokens for Parameter {
//     fn to_tokens(&self, tokens: &mut TokenStream2) {
//         let name = &self.name;
//         let ty = &self.ty;
//         tokens.extend(quote! {
//             (#name, #ty)
//         });
//     }
// }

#[derive(Debug, Clone)]
struct Function {
    name: String,
    parameters: Vec<Parameter>,
    return_type: Option<String>,
    body: Vec<Stmt>,
}

// impl ToTokens for Function {
//     fn to_tokens(&self, tokens: &mut TokenStream2) {
//         let name = &self.name;
//         let params: Vec<TokenStream2> = self
//             .parameters
//             .iter()
//             .map(|p| {
//                 let name = &p.name;
//                 let ty = &p.ty;
//                 quote! { (#name.to_string(), #ty.to_string()) }
//             })
//             .collect();
//
//         let ret_type = self.return_type.as_ref().map(|t| t.to_string());
//         tokens.extend(quote! {
//             wazap_ast::WatFunction {
//                 name: #name.to_string(),
//                 params: vec![#(#params),*],
//                 result: Some(#ret_type.to_string()),
//                 locals: std::collections::HashMap::new(),
//                 locals_counters: std::collections::HashMap::new(),
//                 body: std::collections::Vec::new()
//             }
//         });
//     }
// }

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

        while !input.is_empty() {
            if input.peek(Token![fn]) {
                parse_function(&mut input, &mut functions)?;
            } else if input.peek(Token![type]) {
                let (name, ty) = parse_type_def(&mut input)?;
                types.insert(name, ty);
            } else if input.peek(Token![struct]) {
                let (name, ty) = parse_struct(&mut input)?;
                types.insert(format!("${name}"), ty);
            } else {
                todo!("other input type")
            }
        }

        let mut module = WatModule::new();
        // for now just hardcode the exception tag
        types.insert(
            "$ExceptionType".to_string(),
            WasmType::Func {
                params: vec![WasmType::I32, WasmType::I32],
                result: None,
            },
        );

        module
            .tags
            .insert("$AssertException".to_string(), "$ExceptionType".to_string());

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
            wat_function.body = instructions.into_iter().map(|i| Box::new(i)).collect();
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
    let mut lookahead = input.lookahead1();

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

fn parse_function(input: &mut ParseStream, functions: &mut Vec<Function>) -> Result<()> {
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

    Ok(())
}

fn translate_binary(
    module: &mut WatModule,
    function: &mut WatFunction,
    mut current_block: &mut Vec<WatInstruction>,
    binary: &ExprBinary,
    mut left_instructions: &mut Vec<WatInstruction>,
    mut right_instructions: &mut Vec<WatInstruction>,
) {
    let left_ty = get_type(module, function, left_instructions);
    let right_ty = get_type(module, function, right_instructions);

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

    current_block.append(&mut left_instructions);
    current_block.append(&mut right_instructions);

    let op = match binary.op {
        syn::BinOp::Add(_) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Add,
                WasmType::I64 => WatInstruction::I64Add,
                WasmType::F32 => WatInstruction::F32Add,
                WasmType::F64 => WatInstruction::F64Add,
                WasmType::I8 => WatInstruction::I32Add,
                WasmType::I31Ref => todo!("translate_binary: WasmType::I31Ref"),
                WasmType::Anyref => todo!("translate_binary: WasmType::Anyref "),
                WasmType::Ref(_, _) => todo!("translate_binary: WasmType::Ref(_, _) "),
                WasmType::Array { mutable, ty } => todo!("translate_binary: WasmType::Array(_) "),
                WasmType::Struct(_) => todo!("translate_binary: WasmType::Struct(_) "),
                WasmType::Func { .. } => todo!("translate_binary: WasmType::Struct(_) "),
            };

            current_block.push(instruction);
        }
        syn::BinOp::Sub(_) => todo!("translate_binary: syn::BinOp::Sub(_) "),
        syn::BinOp::Mul(_) => todo!("translate_binary: syn::BinOp::Mul(_) "),
        syn::BinOp::Div(_) => todo!("translate_binary: syn::BinOp::Div(_) "),
        syn::BinOp::Rem(_) => todo!("translate_binary: syn::BinOp::Rem(_) "),
        syn::BinOp::And(_) => todo!("translate_binary: syn::BinOp::And(_) "),
        syn::BinOp::Or(_) => todo!("translate_binary: syn::BinOp::Or(_) "),
        syn::BinOp::BitXor(_) => todo!("translate_binary: syn::BinOp::BitXor(_) "),
        syn::BinOp::BitAnd(_) => todo!("translate_binary: syn::BinOp::BitAnd(_) "),
        syn::BinOp::BitOr(_) => todo!("translate_binary: syn::BinOp::BitOr(_) "),
        syn::BinOp::Shl(_) => todo!("translate_binary: syn::BinOp::Shl(_) "),
        syn::BinOp::Shr(_) => todo!("translate_binary: syn::BinOp::Shr(_) "),
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
                WasmType::Array { mutable, ty } => todo!("translate_binary: WasmType::Array(_) "),
                WasmType::Struct(_) => todo!("translate_binary: WasmType::Struct(_) "),
                WasmType::Func { .. } => todo!("translate_binary: WasmType::Func(_) "),
            };

            current_block.push(instruction);
        }
        syn::BinOp::Lt(_) => todo!("translate_binary: syn::BinOp::Lt(_) "),
        syn::BinOp::Le(_) => todo!("translate_binary: syn::BinOp::Le(_) "),
        syn::BinOp::Ne(_) => todo!("translate_binary: syn::BinOp::Ne(_) "),
        syn::BinOp::Ge(_) => todo!("translate_binary: syn::BinOp::Ge(_) "),
        syn::BinOp::Gt(_) => todo!("translate_binary: syn::BinOp::Gt(_) "),
        syn::BinOp::AddAssign(_) => {}
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
    };
}

fn get_type(
    module: &WatModule,
    function: &WatFunction,
    instructions: &Vec<WatInstruction>,
) -> Option<WasmType> {
    instructions.last().map(|instr| match instr {
        WatInstruction::Nop => todo!("WatInstruction::Local: WatInstruction::Nop "),
        WatInstruction::Local { name, r#type } => {
            todo!("WatInstruction::Local: WatInstruction::Local ")
        }
        WatInstruction::GlobalGet(_) => {
            todo!("WatInstruction::Local: WatInstruction::GlobalGet(_) ")
        }
        // TODO: Handle non existent local
        WatInstruction::LocalGet(name) => function
            .locals
            .get(name)
            .or(function.params.iter().find(|p| &p.0 == name).map(|p| &p.1))
            .ok_or(anyhow!("Could not find local {name}"))
            .unwrap()
            .clone(),
        WatInstruction::LocalSet(_) => todo!("get_type: WatInstruction::LocalSet(_)"),
        WatInstruction::Call(_) => todo!("get_type: WatInstruction::Call(_) "),

        WatInstruction::I32Const(_) => WasmType::I32,
        WatInstruction::I64Const(_) => WasmType::I64,
        WatInstruction::F32Const(_) => WasmType::F32,
        WatInstruction::F64Const(_) => WasmType::F64,

        WatInstruction::I32Eqz => WasmType::I32,
        WatInstruction::I64Eqz => WasmType::I64,
        WatInstruction::F32Eqz => WasmType::F32,
        WatInstruction::F64Eqz => WasmType::F64,

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
        WatInstruction::Block {
            label,
            instructions,
        } => todo!("get_type: WatInstruction::Block "),
        WatInstruction::Loop {
            label,
            instructions,
        } => todo!("get_type: WatInstruction::Loop "),
        WatInstruction::If { then, r#else } => todo!("get_type: WatInstruction::If "),
        WatInstruction::BrIf(_) => todo!("get_type: WatInstruction::BrIf(_) "),
        WatInstruction::Br(_) => todo!("get_type: WatInstruction::Br(_) "),
        WatInstruction::Empty => todo!("get_type: WatInstruction::Empty "),
        WatInstruction::Log => todo!("get_type: WatInstruction::Log "),
        WatInstruction::Identifier(_) => todo!("get_type: WatInstruction::Identifier(_) "),
        WatInstruction::Drop => todo!("get_type: WatInstruction::Drop "),
        WatInstruction::LocalTee(_) => todo!("get_type: WatInstruction::LocalTee(_) "),
        WatInstruction::RefI31 => WasmType::I31Ref,
        WatInstruction::Throw(_) => todo!("get_type: WatInstruction::Throw(_) "),
        WatInstruction::Try {
            try_block,
            catches,
            catch_all,
        } => todo!("get_type: WatInstruction::Try "),
        WatInstruction::Catch(_, _) => todo!("get_type: WatInstruction::Catch(_, "),
        WatInstruction::CatchAll(_) => todo!("get_type: WatInstruction::CatchAll(_) "),
        WatInstruction::I32Add => WasmType::I32,
        WatInstruction::I64Add => WasmType::I64,
        WatInstruction::F32Add => WasmType::F32,
        WatInstruction::F64Add => WasmType::F64,
        WatInstruction::I32GeS => WasmType::I32,
        WatInstruction::ArrayLen => WasmType::I32,
        WatInstruction::ArrayGet(name) => get_element_type(module, function, name).unwrap(),
        WatInstruction::ArrayGetU(name) => WasmType::I32,
        WatInstruction::ArraySet(name) => get_element_type(module, function, name).unwrap(),
        WatInstruction::ArrayNewFixed(typeidx, n) => {
            todo!("get_type: WatInstruction::NewFixed")
        }
        WatInstruction::I32Eq => WasmType::I32,
        WatInstruction::I64Eq => WasmType::I32,
        WatInstruction::F32Eq => WasmType::I32,
        WatInstruction::F64Eq => WasmType::I32,
        WatInstruction::I64ExtendI32S => WasmType::I64,
        WatInstruction::I32WrapI64 => WasmType::I32,
        WatInstruction::I31GetS => WasmType::I32,
        WatInstruction::F64PromoteF32 => WasmType::F64,
        WatInstruction::F32DemoteF64 => WasmType::F32,
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
    module: &WatModule,
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
    module: &WatModule,
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
    function: &WatFunction,
    instructions: &Vec<WatInstruction>,
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
    function: &WatFunction,
    instructions: &Vec<WatInstruction>,
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
    function: &WatFunction,
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
    mut module: &mut WatModule,
    mut function: &mut WatFunction,
    mut current_block: &mut Vec<WatInstruction>,
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
    mut module: &mut WatModule,
    mut function: &mut WatFunction,
    mut current_block: &mut Vec<WatInstruction>,
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
                    println!("assign local_type: {local_type:#?}");
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
            let mut left_instructions = Vec::new();
            let mut right_instructions = Vec::new();
            translate_expression(
                module,
                function,
                &mut left_instructions,
                binary.left.deref(),
                None,
                None,
            );
            translate_expression(
                module,
                function,
                &mut right_instructions,
                binary.right.deref(),
                None,
                None,
            );

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
        Expr::Block(_) => todo!("translate_expression: Expr::Block(_) "),
        Expr::Break(_) => todo!("translate_expression: Expr::Break(_) "),
        Expr::Call(expr_call) => {
            if let Expr::Path(syn::ExprPath { path, .. }) = expr_call.func.deref() {
                let func_name = path.segments[0].ident.to_string();

                println!("func_name: {func_name}");
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

            let mut i = 0;
            // TODO: for some reason .enumerate() doesn't work here, I don't have time to
            // investigate
            for field in &expr_struct.fields {
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
                i += 1;
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
            WasmType::I32 => quote! { wazap_ast::WasmType::I32 },
            WasmType::I64 => quote! { wazap_ast::WasmType::I64 },
            WasmType::F32 => quote! { wazap_ast::WasmType::F32 },
            WasmType::F64 => quote! { wazap_ast::WasmType::F64 },
            WasmType::I8 => quote! { wazap_ast::WasmType::I8 },
            WasmType::I31Ref => quote! { wazap_ast::WasmType::I31Ref },
            WasmType::Anyref => quote! { wazap_ast::WasmType::Anyref },
            WasmType::Ref(r, nullable) => {
                let nullable = match nullable {
                    wazap_ast::Nullable::True => quote! {wazap_ast::Nullable::True},
                    wazap_ast::Nullable::False => quote! { wazap_ast::Nullable::False },
                };
                quote! { wazap_ast::WasmType::Ref(#r.to_string(), #nullable) }
            }
            WasmType::Array { mutable, ty } => {
                let ty = OurWasmType(*ty.clone());
                quote! { wazap_ast::WasmType::Array { mutable: #mutable, ty: Box::new(#ty) } }
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

                    quote! { wazap_ast::StructField { name: #name, ty: #ty, mutable: #mutable } }
                });
                quote! {
                    wazap_ast::WasmType::Struct(vec![#(#fields),*])
                }
            }
            WasmType::Func { params, result } => {
                let result = if let Some(result) = result {
                    let result = OurWasmType(result.deref().clone());
                    quote! { Some(#result) }
                } else {
                    quote! { None }
                };
                let params = params.iter().map(|param| {
                    let param = OurWasmType(param.clone());
                    quote! { #param }
                });

                quote! {
                    wazap_ast::WasmType::Func { params: vec![#(#params),*], result: #result }
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
            WatInstruction::Local { name, r#type } => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Local ")
            }
            WatInstruction::GlobalGet(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::GlobalGet(_) ")
            }
            WatInstruction::LocalGet(name) => {
                quote! { wazap_ast::WatInstruction::LocalGet(#name.to_string()) }
            }
            WatInstruction::LocalSet(name) => {
                quote! { wazap_ast::WatInstruction::LocalSet(#name.to_string()) }
            }
            WatInstruction::Call(name) => {
                quote! { wazap_ast::WatInstruction::Call(#name.to_string()) }
            }
            WatInstruction::I32Const(value) => {
                quote! { wazap_ast::WatInstruction::I32Const(#value) }
            }
            WatInstruction::I64Const(value) => {
                quote! { wazap_ast::WatInstruction::I32Const(#value) }
            }
            WatInstruction::F32Const(value) => {
                quote! { wazap_ast::WatInstruction::F32Const(#value) }
            }
            WatInstruction::F64Const(value) => {
                quote! { wazap_ast::WatInstruction::F64Const(#value) }
            }

            WatInstruction::I32Eqz => quote! { wazap_ast::WatInstruction::I32Eqz },
            WatInstruction::I64Eqz => quote! { wazap_ast::WatInstruction::I64Eqz },
            WatInstruction::F32Eqz => quote! { wazap_ast::WatInstruction::F32Eqz },
            WatInstruction::F64Eqz => quote! { wazap_ast::WatInstruction::F64Eqz },

            WatInstruction::StructNew(type_name) => {
                quote! { wazap_ast::WatInstruction::StructNew(#type_name.to_string()) }
            }
            WatInstruction::StructGet(type_name, field_name) => {
                quote! { wazap_ast::WatInstruction::StructGet(#type_name.to_string(), #field_name.to_string() ) }
            }
            WatInstruction::StructSet(type_name, field_name) => {
                quote! { wazap_ast::WatInstruction::StructSet(#type_name.to_string(), #field_name.to_string() ) }
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
            WatInstruction::Return => quote! { wazap_ast::WatInstruction::Return },
            WatInstruction::ReturnCall(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::ReturnCall(_) ")
            }
            WatInstruction::Block {
                label,
                instructions,
            } => {
                let instructions = instructions.iter().map(|i| OurWatInstruction(i.clone()));
                quote! {
                    wazap_ast::WatInstruction::block(#label, vec![#(#instructions),*])
                }
            }
            WatInstruction::Loop {
                label,
                instructions,
            } => {
                let instructions = instructions.iter().map(|i| OurWatInstruction(i.clone()));
                quote! {
                    wazap_ast::WatInstruction::r#loop(#label, vec![#(#instructions),*])
                }
            }
            WatInstruction::If { then, r#else } => {
                let then_instructions = then.iter().map(|i| OurWatInstruction(i.clone()));
                let else_code = if let Some(r#else) = r#else {
                    let else_instructions = r#else.iter().map(|i| OurWatInstruction(i.clone()));
                    quote! { vec![#(#else_instructions),*] }
                } else {
                    quote! { None }
                };

                quote! {
                    wazap_ast::WatInstruction::If {then: vec![#(#then_instructions),*], r#else: #else_code }
                }
            }
            WatInstruction::BrIf(label) => {
                quote! { wazap_ast::WatInstruction::br_if(#label) }
            }
            WatInstruction::Br(label) => {
                quote! { wazap_ast::WatInstruction::br(#label) }
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
                quote! { wazap_ast::WatInstruction::LocalTee(#name.to_string()) }
            }
            WatInstruction::RefI31 => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::RefI31(_) ")
            }
            WatInstruction::Throw(label) => {
                quote! { wazap_ast::WatInstruction::Throw(#label.to_string()) }
            }
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => todo!("impl ToTokens for OurWatInstruction: WatInstruction::Try "),
            WatInstruction::Catch(_, _) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Catch(_, ")
            }
            WatInstruction::CatchAll(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::CatchAll(_) ")
            }
            WatInstruction::I32Add => quote! { wazap_ast::WatInstruction::I32Add },
            WatInstruction::I64Add => quote! { wazap_ast::WatInstruction::I64Add },
            WatInstruction::F32Add => quote! { wazap_ast::WatInstruction::F32Add },
            WatInstruction::F64Add => quote! { wazap_ast::WatInstruction::F64Add },
            WatInstruction::I32GeS => quote! { wazap_ast::WatInstruction::I32GeS },
            WatInstruction::ArrayLen => quote! { wazap_ast::WatInstruction::ArrayLen },
            WatInstruction::ArrayGet(name) => {
                quote! { wazap_ast::WatInstruction::ArrayGet(#name.to_string()) }
            }
            WatInstruction::ArrayGetU(name) => {
                quote! { wazap_ast::WatInstruction::ArrayGetU(#name.to_string()) }
            }
            WatInstruction::ArraySet(name) => {
                quote! { wazap_ast::WatInstruction::ArraySet(#name.to_string()) }
            }
            WatInstruction::ArrayNewFixed(typeidx, n) => {
                quote! { wazap_ast::WatInstruction::ArrayNewFixed(#typeidx.to_string(), #n) }
            }
            WatInstruction::I32Eq => quote! { wazap_ast::WatInstruction::I32Eq },
            WatInstruction::I64Eq => quote! { wazap_ast::WatInstruction::I64Eq },
            WatInstruction::F32Eq => quote! { wazap_ast::WatInstruction::F32Eq },
            WatInstruction::F64Eq => quote! { wazap_ast::WatInstruction::F64Eq },
            WatInstruction::I64ExtendI32S => quote! { wazap_ast::WatInstruction::I64ExtendI32S },
            WatInstruction::I32WrapI64 => quote! { wazap_ast::WatInstruction::I32WrapI64 },
            WatInstruction::I31GetS => quote! { wazap_ast::WatInstruction::I31GetS },
            WatInstruction::F64PromoteF32 => quote! { wazap_ast::WatInstruction::F64PromoteF32 },
            WatInstruction::F32DemoteF64 => quote! { wazap_ast::WatInstruction::F32DemoteF64 },
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
            let mut function = wazap_ast::WatFunction::new(#name);

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
                    Some(WasmType::Ref(ty, wazap_ast::Nullable::True))
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
    function: &mut WatFunction,
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

                println!("local.init {:#?}\n{ty:#?}", local.init);
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

    let functions = global_scope.functions.into_iter().map(|f| {
        let our = OurWatFunction(f.clone());
        quote! {
            #our

            module.add_function(function);
        }
    });

    let output = quote! {
        {
            let mut module = wazap_ast::WatModule::new();

            #(#data)*

            #(#types)*

            #(#tags)*

            #(#functions)*

            module
        }
    };
    println!("output:\n{}", output);

    output.into()
}

fn map_type(ty: &str) -> String {
    match ty {
        "i32" => "i32".to_string(),
        "i64" => "i64".to_string(),
        "f32" => "f32".to_string(),
        "f64" => "f64".to_string(),
        _ => panic!("Unsupported type: {}", ty),
    }
}
