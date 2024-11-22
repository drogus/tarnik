use proc_macro::TokenStream;
use std::collections::HashMap;
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    token::{self, Semi},
    Expr, ExprBinary, PatType, Type,
};

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{braced, parenthesized, parse_macro_input, Ident, Result, Stmt, Token};

use wazap_ast::{self as ast, WasmType, WatFunction, WatInstruction};

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
    functions: Vec<Function>,
    types: Vec<(String, WasmType)>,
}

impl Parse for GlobalScope {
    fn parse(mut input: ParseStream) -> Result<Self> {
        let mut functions = Vec::new();
        let mut types = Vec::new();

        while !input.is_empty() {
            if input.peek(Token![fn]) {
                parse_function(&mut input, &mut functions).unwrap();
            } else if input.peek(Token![type]) {
                parse_type(&mut input, &mut types).unwrap();
            } else {
                todo!("other input type")
            }
        }

        Ok(Self { functions, types })
    }
}

fn parse_type(input: &mut ParseStream, types: &mut Vec<(String, WasmType)>) -> Result<()> {
    let _: Token![type] = input.parse()?;

    let ident: Ident = input.parse()?;

    let _: Token![=] = input.parse()?;

    if input.peek(token::Bracket) {
        // custom parsing for array types
        let content;
        bracketed!(content in input);

        let mutable = content.parse::<Token![mut]>().is_ok();
        let rust_type: Type = content.parse()?;
        let ty = Box::new(translate_type(&rust_type).unwrap());
        let name = format!("${ident}");
        types.push((name, WasmType::Array { mutable, ty }));
    } else {
        let ty: syn::Type = input.parse()?;
    }

    let _: token::Semi = input.parse()?;

    Ok(())
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
    function: &mut WatFunction,
    mut current_block: &mut Vec<WatInstruction>,
    binary: &ExprBinary,
    ty: WasmType,
) {
    let op = match binary.op {
        syn::BinOp::Add(_) => {
            let instruction = match ty {
                WasmType::I32 => WatInstruction::I32Add,
                WasmType::I64 => WatInstruction::I64Add,
                WasmType::F32 => WatInstruction::F32Add,
                WasmType::F64 => WatInstruction::F64Add,
                WasmType::I31Ref => todo!("translate_binary: WasmType::I31Ref"),
                WasmType::Anyref => todo!("translate_binary: WasmType::Anyref "),
                WasmType::Ref(_, _) => todo!("translate_binary: WasmType::Ref(_, _) "),
                WasmType::Array { mutable, ty } => todo!("translate_binary: WasmType::Array(_) "),
                WasmType::Struct(_) => todo!("translate_binary: WasmType::Struct(_) "),
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
        syn::BinOp::Eq(_) => todo!("translate_binary: syn::BinOp::Eq(_) "),
        syn::BinOp::Lt(_) => todo!("translate_binary: syn::BinOp::Lt(_) "),
        syn::BinOp::Le(_) => todo!("translate_binary: syn::BinOp::Le(_) "),
        syn::BinOp::Ne(_) => todo!("translate_binary: syn::BinOp::Ne(_) "),
        syn::BinOp::Ge(_) => todo!("translate_binary: syn::BinOp::Ge(_) "),
        syn::BinOp::Gt(_) => todo!("translate_binary: syn::BinOp::Gt(_) "),
        syn::BinOp::AddAssign(_) => todo!("translate_binary: syn::BinOp::AddAssign(_) "),
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

fn get_type(function: &WatFunction, instructions: &Vec<WatInstruction>) -> Option<WasmType> {
    instructions
        .last()
        .map(|instr| match instr {
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
                .unwrap(),
            WatInstruction::LocalSet(_) => todo!("get_type: WatInstruction::LocalSet(_)"),
            WatInstruction::Call(_) => todo!("get_type: WatInstruction::Call(_) "),

            WatInstruction::I32Const(_) => &WasmType::I32,
            WatInstruction::I64Const(_) => &WasmType::I64,
            WatInstruction::F32Const(_) => &WasmType::F32,
            WatInstruction::F64Const(_) => &WasmType::F64,

            WatInstruction::I32Eqz => &WasmType::I32,
            WatInstruction::I64Eqz => &WasmType::I64,
            WatInstruction::F32Eqz => &WasmType::F32,
            WatInstruction::F64Eqz => &WasmType::F64,

            WatInstruction::StructNew(_) => todo!("get_type: WatInstruction::StructNew(_) "),
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
            WatInstruction::If {
                condition,
                then,
                r#else,
            } => todo!("get_type: WatInstruction::If "),
            WatInstruction::BrIf(_) => todo!("get_type: WatInstruction::BrIf(_) "),
            WatInstruction::Br(_) => todo!("get_type: WatInstruction::Br(_) "),
            WatInstruction::Empty => todo!("get_type: WatInstruction::Empty "),
            WatInstruction::Log => todo!("get_type: WatInstruction::Log "),
            WatInstruction::Identifier(_) => todo!("get_type: WatInstruction::Identifier(_) "),
            WatInstruction::Drop => todo!("get_type: WatInstruction::Drop "),
            WatInstruction::LocalTee(_) => todo!("get_type: WatInstruction::LocalTee(_) "),
            WatInstruction::RefI31(_) => todo!("get_type: WatInstruction::RefI31(_) "),
            WatInstruction::Throw(_) => todo!("get_type: WatInstruction::Throw(_) "),
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => todo!("get_type: WatInstruction::Try "),
            WatInstruction::Catch(_, _) => todo!("get_type: WatInstruction::Catch(_, "),
            WatInstruction::CatchAll(_) => todo!("get_type: WatInstruction::CatchAll(_) "),
            WatInstruction::I32Add => &WasmType::I32,
            WatInstruction::I64Add => &WasmType::I64,
            WatInstruction::F32Add => &WasmType::F32,
            WatInstruction::F64Add => &WasmType::F64,
        })
        .cloned()
}

fn translate_expression(
    mut function: &mut WatFunction,
    mut current_block: &mut Vec<WatInstruction>,
    expr: &Expr,
    _: &Option<Semi>,
) {
    match expr {
        Expr::Array(_) => todo!("translate_expression: Expr::Array(_) "),
        Expr::Assign(_) => todo!("translate_expression: Expr::Assign(_) "),
        Expr::Async(_) => todo!("translate_expression: Expr::Async(_) "),
        Expr::Await(_) => todo!("translate_expression: Expr::Await(_) "),
        Expr::Binary(binary) => {
            translate_expression(&mut function, &mut current_block, &*binary.left, &None);
            let left_ty = get_type(&function, &current_block);
            translate_expression(&mut function, &mut current_block, &*binary.right, &None);
            let right_ty = get_type(&function, &current_block);

            // TODO: handle casts and/or error handling

            translate_binary(&mut function, &mut current_block, binary, left_ty.unwrap());
        }
        Expr::Block(_) => todo!("translate_expression: Expr::Block(_) "),
        Expr::Break(_) => todo!("translate_expression: Expr::Break(_) "),
        Expr::Call(_) => todo!("translate_expression: Expr::Call(_) "),
        Expr::Cast(_) => todo!("translate_expression: Expr::Cast(_) "),
        Expr::Closure(_) => todo!("translate_expression: Expr::Closure(_) "),
        Expr::Const(_) => todo!("translate_expression: Expr::Const(_) "),
        Expr::Continue(_) => todo!("translate_expression: Expr::Continue(_) "),
        Expr::Field(_) => todo!("translate_expression: Expr::Field(_) "),
        Expr::ForLoop(_) => todo!("translate_expression: Expr::ForLoop(_) "),
        Expr::Group(_) => todo!("translate_expression: Expr::Group(_) "),
        Expr::If(_) => todo!("translate_expression: Expr::If(_) "),
        Expr::Index(_) => todo!("translate_expression: Expr::Index(_) "),
        Expr::Infer(_) => todo!("translate_expression: Expr::Infer(_) "),
        Expr::Let(_) => todo!("translate_expression: Expr::Let(_) "),
        Expr::Lit(_) => todo!("translate_expression: Expr::Lit(_) "),
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
                translate_expression(function, current_block, expr, &None)
            }
            current_block.push(WatInstruction::Return);
        }
        Expr::Struct(_) => todo!("translate_expression: Expr::Struct(_) "),
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
                let fields: Vec<OurWasmType> = s.iter().map(|ty| OurWasmType(ty.clone())).collect();
                quote! { wazap_ast::WasmType::Struct(#(#fields),*) }
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

            WatInstruction::StructNew(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::StructNew(_) ")
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
            } => todo!("impl ToTokens for OurWatInstruction: WatInstruction::Block "),
            WatInstruction::Loop {
                label,
                instructions,
            } => todo!("impl ToTokens for OurWatInstruction: WatInstruction::Loop "),
            WatInstruction::If {
                condition,
                then,
                r#else,
            } => todo!("impl ToTokens for OurWatInstruction: WatInstruction::If "),
            WatInstruction::BrIf(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::BrIf(_) ")
            }
            WatInstruction::Br(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Br(_) ")
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
            WatInstruction::LocalTee(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::LocalTee(_) ")
            }
            WatInstruction::RefI31(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::RefI31(_) ")
            }
            WatInstruction::Throw(_) => {
                todo!("impl ToTokens for OurWatInstruction: WatInstruction::Throw(_) ")
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

#[proc_macro]
pub fn wasm(input: TokenStream) -> TokenStream {
    let global_scope = parse_macro_input!(input as GlobalScope);

    // Generate Rust code that builds the WAT module
    let functions = global_scope.functions.iter().map(|f| {
        let mut wat_function = WatFunction::new(&f.name);
        for param in &f.parameters {
            wat_function.add_param(format!("${}", param.name), &param.ty);
        }
        for stmt in &f.body {
            let mut instructions = Vec::new();
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
                        let (name, ty) = translate_pat_type(&mut wat_function, pat_type);
                        let ty = ty.unwrap(); // type should be known at this point
                        wat_function.add_local_exact(name, ty);
                    }
                    syn::Pat::Verbatim(_) => todo!("Stmt::Local(local): syn::Pat::Verbatim(_) "),
                    syn::Pat::Wild(_) => todo!("Stmt::Local(local): syn::Pat::Wild(_) "),
                    _ => todo!("Stmt::Local(local): _ "),
                },
                Stmt::Item(_) => todo!("Stmt::Item(_) "),
                Stmt::Expr(expr, semi) => {
                    translate_expression(&mut wat_function, &mut instructions, expr, semi)
                }
                Stmt::Macro(_) => todo!("Stmt::Macro(_) "),
            }

            wat_function.body = instructions.into_iter().map(|i| Box::new(i)).collect();
        }

        let our = OurWatFunction(wat_function);
        quote! {
            #our

            module.add_function(function);
        }
    });

    let types = global_scope.types.into_iter().map(|(name, ty)| {
        let ty = OurWasmType(ty);
        quote! {
            module.add_type(#name.to_string(), #ty);
        }
    });

    let output = quote! {
        {
            let mut module = wazap_ast::WatModule::new();

            #(#types)*

            #(#functions)*

            module
        }
    };
    println!("output:\n{}", output.to_string());

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
