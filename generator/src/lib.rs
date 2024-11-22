use proc_macro::TokenStream;
use std::collections::HashMap;
use syn::{
    parse::{Parse, ParseStream},
    token::Semi,
    Expr, ExprBinary,
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
}

impl Parse for GlobalScope {
    fn parse(mut input: ParseStream) -> Result<Self> {
        let mut functions = Vec::new();

        while !input.is_empty() {
            if input.peek(Token![fn]) {
                parse_function(&mut input, &mut functions);
            }
        }

        Ok(Self { functions })
    }
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
                WasmType::I31Ref => todo!(),
                WasmType::Anyref => todo!(),
                WasmType::Ref(_) => todo!(),
                WasmType::Array(_) => todo!(),
                WasmType::Struct(_) => todo!(),
            };

            current_block.push(instruction);
        }
        syn::BinOp::Sub(_) => todo!(),
        syn::BinOp::Mul(_) => todo!(),
        syn::BinOp::Div(_) => todo!(),
        syn::BinOp::Rem(_) => todo!(),
        syn::BinOp::And(_) => todo!(),
        syn::BinOp::Or(_) => todo!(),
        syn::BinOp::BitXor(_) => todo!(),
        syn::BinOp::BitAnd(_) => todo!(),
        syn::BinOp::BitOr(_) => todo!(),
        syn::BinOp::Shl(_) => todo!(),
        syn::BinOp::Shr(_) => todo!(),
        syn::BinOp::Eq(_) => todo!(),
        syn::BinOp::Lt(_) => todo!(),
        syn::BinOp::Le(_) => todo!(),
        syn::BinOp::Ne(_) => todo!(),
        syn::BinOp::Ge(_) => todo!(),
        syn::BinOp::Gt(_) => todo!(),
        syn::BinOp::AddAssign(_) => todo!(),
        syn::BinOp::SubAssign(_) => todo!(),
        syn::BinOp::MulAssign(_) => todo!(),
        syn::BinOp::DivAssign(_) => todo!(),
        syn::BinOp::RemAssign(_) => todo!(),
        syn::BinOp::BitXorAssign(_) => todo!(),
        syn::BinOp::BitAndAssign(_) => todo!(),
        syn::BinOp::BitOrAssign(_) => todo!(),
        syn::BinOp::ShlAssign(_) => todo!(),
        syn::BinOp::ShrAssign(_) => todo!(),
        _ => todo!(),
    };
}

fn get_type(function: &WatFunction, instructions: &Vec<WatInstruction>) -> Option<WasmType> {
    instructions
        .last()
        .map(|instr| match instr {
            WatInstruction::Nop => todo!(),
            WatInstruction::Local { name, r#type } => todo!(),
            WatInstruction::GlobalGet(_) => todo!(),
            // TODO: Handle non existent local
            WatInstruction::LocalGet(name) => {
                println!("{function:#?}");
                println!("LOCAL GET {name}");
                function
                    .locals
                    .get(name)
                    .or(function.params.iter().find(|p| &p.0 == name).map(|p| &p.1))
                    .unwrap()
            }
            WatInstruction::LocalSet(_) => todo!(),
            WatInstruction::Call(_) => todo!(),

            WatInstruction::I32Const(_) => &WasmType::I32,
            WatInstruction::I64Const(_) => &WasmType::I64,
            WatInstruction::F32Const(_) => &WasmType::F32,
            WatInstruction::F64Const(_) => &WasmType::F64,

            WatInstruction::I32Eqz => &WasmType::I32,
            WatInstruction::I64Eqz => &WasmType::I64,
            WatInstruction::F32Eqz => &WasmType::F32,
            WatInstruction::F64Eqz => &WasmType::F64,

            WatInstruction::StructNew(_) => todo!(),
            WatInstruction::ArrayNew(_) => todo!(),
            WatInstruction::RefNull(_) => todo!(),
            WatInstruction::Ref(_) => todo!(),
            WatInstruction::RefFunc(_) => todo!(),
            WatInstruction::Type(_) => todo!(),
            WatInstruction::Return => todo!(),
            WatInstruction::ReturnCall(_) => todo!(),
            WatInstruction::Block {
                label,
                instructions,
            } => todo!(),
            WatInstruction::Loop {
                label,
                instructions,
            } => todo!(),
            WatInstruction::If {
                condition,
                then,
                r#else,
            } => todo!(),
            WatInstruction::BrIf(_) => todo!(),
            WatInstruction::Br(_) => todo!(),
            WatInstruction::Empty => todo!(),
            WatInstruction::Log => todo!(),
            WatInstruction::Identifier(_) => todo!(),
            WatInstruction::Drop => todo!(),
            WatInstruction::LocalTee(_) => todo!(),
            WatInstruction::RefI31(_) => todo!(),
            WatInstruction::Throw(_) => todo!(),
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => todo!(),
            WatInstruction::Catch(_, _) => todo!(),
            WatInstruction::CatchAll(_) => todo!(),
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
        Expr::Array(_) => todo!(),
        Expr::Assign(_) => todo!(),
        Expr::Async(_) => todo!(),
        Expr::Await(_) => todo!(),
        Expr::Binary(binary) => {
            translate_expression(&mut function, &mut current_block, &*binary.left, &None);
            let left_ty = get_type(&function, &current_block);
            translate_expression(&mut function, &mut current_block, &*binary.right, &None);
            let right_ty = get_type(&function, &current_block);

            // TODO: handle casts and/or error handling

            translate_binary(&mut function, &mut current_block, binary, left_ty.unwrap());
        }
        Expr::Block(_) => todo!(),
        Expr::Break(_) => todo!(),
        Expr::Call(_) => todo!(),
        Expr::Cast(_) => todo!(),
        Expr::Closure(_) => todo!(),
        Expr::Const(_) => todo!(),
        Expr::Continue(_) => todo!(),
        Expr::Field(_) => todo!(),
        Expr::ForLoop(_) => todo!(),
        Expr::Group(_) => todo!(),
        Expr::If(_) => todo!(),
        Expr::Index(_) => todo!(),
        Expr::Infer(_) => todo!(),
        Expr::Let(_) => todo!(),
        Expr::Lit(_) => todo!(),
        Expr::Loop(_) => todo!(),
        Expr::Macro(_) => todo!(),
        Expr::Match(_) => todo!(),
        Expr::MethodCall(_) => todo!(),
        Expr::Paren(_) => todo!(),
        Expr::Path(path_expr) => {
            let name = path_expr.path.segments[0].ident.to_string();
            current_block.push(WatInstruction::LocalGet(format!("${name}")));
        }
        Expr::Range(_) => todo!(),
        Expr::RawAddr(_) => todo!(),
        Expr::Reference(_) => todo!(),
        Expr::Repeat(_) => todo!(),
        Expr::Return(ret) => {
            if let Some(expr) = &ret.expr {
                translate_expression(function, current_block, expr, &None)
            }
            current_block.push(WatInstruction::Return);
        }
        Expr::Struct(_) => todo!(),
        Expr::Try(_) => todo!(),
        Expr::TryBlock(_) => todo!(),
        Expr::Tuple(_) => todo!(),
        Expr::Unary(_) => todo!(),
        Expr::Unsafe(_) => todo!(),
        Expr::Verbatim(_) => todo!(),
        Expr::While(_) => todo!(),
        Expr::Yield(_) => todo!(),
        _ => todo!(),
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
            WasmType::Ref(r) => quote! { wazap_ast::WasmType::Ref(#r.to_string()) },
            WasmType::Array(ty) => {
                let ty = OurWasmType(*ty.clone());
                quote! { wazap_ast::WasmType::Array(#ty) }
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
            WatInstruction::Nop => todo!(),
            WatInstruction::Local { name, r#type } => todo!(),
            WatInstruction::GlobalGet(_) => todo!(),
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

            WatInstruction::StructNew(_) => todo!(),
            WatInstruction::ArrayNew(_) => todo!(),
            WatInstruction::RefNull(_) => todo!(),
            WatInstruction::Ref(_) => todo!(),
            WatInstruction::RefFunc(_) => todo!(),
            WatInstruction::Type(_) => todo!(),
            WatInstruction::Return => quote! { wazap_ast::WatInstruction::Return },
            WatInstruction::ReturnCall(_) => todo!(),
            WatInstruction::Block {
                label,
                instructions,
            } => todo!(),
            WatInstruction::Loop {
                label,
                instructions,
            } => todo!(),
            WatInstruction::If {
                condition,
                then,
                r#else,
            } => todo!(),
            WatInstruction::BrIf(_) => todo!(),
            WatInstruction::Br(_) => todo!(),
            WatInstruction::Empty => todo!(),
            WatInstruction::Log => todo!(),
            WatInstruction::Identifier(_) => todo!(),
            WatInstruction::Drop => todo!(),
            WatInstruction::LocalTee(_) => todo!(),
            WatInstruction::RefI31(_) => todo!(),
            WatInstruction::Throw(_) => todo!(),
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => todo!(),
            WatInstruction::Catch(_, _) => todo!(),
            WatInstruction::CatchAll(_) => todo!(),
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

        tokens.extend(quote! {
            let mut function = wazap_ast::WatFunction::new(#name);

            #(#params);*;
            #(#instructions);*;
        });
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
            println!("Statmenet: {stmt:#?}");
            match stmt {
                Stmt::Local(_) => todo!(),
                Stmt::Item(_) => todo!(),
                Stmt::Expr(expr, semi) => {
                    translate_expression(&mut wat_function, &mut instructions, expr, semi)
                }
                Stmt::Macro(_) => todo!(),
            }

            wat_function.body = instructions.into_iter().map(|i| Box::new(i)).collect();
        }

        let our = OurWatFunction(wat_function);
        println!("{our:#?}");
        quote! {
            #our

            module.add_function(function);
        }
    });
    let output = quote! {
        {
            let mut module = wazap_ast::WatModule::new();

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
