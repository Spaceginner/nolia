pub mod lcr;
pub(super) mod conversions;
pub mod path;

use std::collections::HashMap;
use rand::distr::SampleString;
use crate::parser::ast;
use super::Compiler;


pub struct CrateSource<'s> {
    pub id: (Box<str>, (u16, u16, u16)),
    pub mods: Vec<(lcr::Path, ast::Module<'s>)>,
    pub deps: Vec<(Box<str>, (u16, u16, u16))>,
}

impl Compiler {
    pub fn load_crate(&mut self, crate_: CrateSource<'_>) {
        let mut impls = HashMap::new();
        let mut funcs = HashMap::new();
        let mut meths = HashMap::new();
        let mut items = HashMap::new();

        for (root, r#mod) in crate_.mods {
            for decl in r#mod.decls {
                let p = root.clone();
                match decl {
                    ast::Declaration::Include { item } => { todo!("includes not supported"); },
                    ast::Declaration::Data { .. } => { todo!("data not supported"); },
                    ast::Declaration::Protocol { .. } => { todo!("protos not supported"); },
                    ast::Declaration::Implementation { .. } => { todo!("impls not supported"); },
                    ast::Declaration::Function { of, r#impl, name, closure } => {
                        assert!(r#impl.is_none(), "impls not supported");
                        assert!(of.is_none(), "methods not supported");

                        let f_type = lcr::FunctionType {
                            captures: closure.sig.captures.into_iter().map(|c| (c.name.into(), transform_type(&p, c.r#type))).collect(),
                            r#return: closure.sig.r#return.map(|r| transform_type(&p, r)).unwrap_or(lcr::TypeRef::Nothing),
                            generics: closure.sig.generics.defs.into_iter().map(|def| (def.name.into(), def.constraint.map(|g| transform_type(&p, g)))).collect(),
                            errors: closure.sig.errors.into_iter().map(|i| transform_item(&p, i).into()).collect(),
                        };

                        let code = closure.code.either(
                            |stmt_b| lcr::Block::Structured(transform_stmt_b(&p, stmt_b)),
                            |asm_b| lcr::Block::Asm(transform_asm_b(&p, asm_b)),
                        );

                        funcs.insert(
                            p.extend(name),
                            lcr::Function { r#type: f_type, code },
                        );
                    },
                    ast::Declaration::Const { what: ast::Capture { name, .. }, val } => {
                        // fixme type is ignored
                        items.insert(
                            p.extend(name),
                            val.into(),
                        );
                    },
                };
            };
        };

        self.crates.insert(
            crate_.id,
            lcr::Crate {
                deps: crate_.deps,
                implementation_store: impls,
                method_store: meths,
                function_store: funcs,
                item_store: items,
            },
        );
    }
}

fn transform_stmt_b(root: &lcr::Path, stmt_b: ast::StatementBlock<'_>) -> lcr::SBlock {
    let mut decls = Vec::new();
    let code = stmt_b.code.into_iter()
        .filter_map(|stmt| {
            Some(match stmt {
                ast::Statement::Declaration { what, with } => {
                    decls.push(lcr::Declaration {
                        name: what.name.into(),
                        r#type: transform_type(root, what.r#type),
                    });
                    with.map(|e| lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment {
                        what: lcr::SBlock { label: None, tag: Box::new(lcr::SBlockTag::Simple {
                            closed: false,
                            decls: vec![],
                            code: vec![lcr::Instruction::DoAction(lcr::ActionInstruction::Load {
                                item: lcr::IntermediatePath {
                                    var: Some(what.name.into()),
                                    path: lcr::Path { crate_: None, parts: vec![] },
                                },
                            })],
                        }) },
                        to: transform_expr(root, e),
                    }))?
                },
                ast::Statement::Eval { expr } =>
                    lcr::Instruction::DoBlock(transform_expr(root, expr)),
                ast::Statement::Assignment { what, to } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment {
                        what: transform_expr(root, what),
                        to: transform_expr(root, to),
                    }),
                ast::Statement::Escape { value, label } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape {
                        value: value.map(|e| transform_expr(root, e)),
                        label: label.into(),
                    }),
                ast::Statement::Repeat { label } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat {
                        label: label.into(),
                    }),
                ast::Statement::Return { value } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Return {
                        value: value.map(|e| transform_expr(root, e)),
                    }),
                ast::Statement::Throw { error } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw {
                        error: transform_expr(root, error)
                    }),
                ast::Statement::VmDebug { dcode } =>
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { dcode })
            })
        }).collect();

    lcr::SBlock {
        label: None,
        tag: Box::new(lcr::SBlockTag::Simple {
            code, decls, closed: stmt_b.closed
        }),
    }
}

fn transform_expr(root: &lcr::Path, expr: ast::Expression<'_>) -> lcr::SBlock {
    match expr {
        ast::Expression::Action(act_e) =>
            lcr::SBlock {
                tag: Box::new(lcr::SBlockTag::Simple {
                    closed: false,
                    decls: vec![],
                    code: vec![lcr::Instruction::DoAction(match *act_e {
                        ast::ActionExpression::Call { what, args } =>
                            lcr::ActionInstruction::Call {
                                what: transform_expr(root, what),
                                args: args.into_iter().map(|e| transform_expr(root, e)).collect(),
                            },
                        ast::ActionExpression::MethodCall { what, method, args } =>
                            lcr::ActionInstruction::MethodCall {
                                what: transform_expr(root, what),
                                method: method.into(),
                                args: args.into_iter().map(|e| transform_expr(root, e)).collect()
                            },
                        ast::ActionExpression::Access { of, field } =>
                            lcr::ActionInstruction::Access {
                                of: transform_expr(root, of),
                                field: field.into(),
                            },
                        ast::ActionExpression::Load { item } =>
                            lcr::ActionInstruction::Load {
                                item: transform_im_item(root, item),
                            },
                    })],
                }),
                label: None,
            },
        ast::Expression::Block(block_e) => {
            let tag = match *block_e.kind {
                ast::BlockExpressionKind::Simple { code } => *transform_stmt_b(root, code).tag,
                ast::BlockExpressionKind::Condition { code, check, otherwise, inverted } =>
                    lcr::SBlockTag::Selector {
                        of: transform_expr(root, check),
                        cases: vec![(
                                lcr::SBlock {
                                    label: None,
                                    tag: Box::new(lcr::SBlockTag::Simple {
                                        closed: false,
                                        decls: vec![],
                                        code: vec![lcr::Instruction::LoadLiteral(ast::LiteralExpression::Bool(!inverted).into())],
                                    }),
                                },
                                transform_stmt_b(root, code)
                            )],
                        fallback: otherwise.map(|o| transform_stmt_b(root, o)),
                    },
                ast::BlockExpressionKind::Selector { of, fallback, cases } =>
                    lcr::SBlockTag::Selector {
                        of: transform_expr(root, of),
                        cases: cases.into_iter()
                            .map(|(e, sb)| (transform_expr(root, e), transform_stmt_b(root, sb)))
                            .collect(),
                        fallback: fallback.map(|sb| transform_stmt_b(root, sb)),
                    },
                ast::BlockExpressionKind::Handle { of, handlers, fallback } =>
                    lcr::SBlockTag::Handle {
                        what: transform_expr(root, of),
                        handlers: handlers.into_iter()
                            .map(|(r#type, name, sb)|
                            (transform_item(root, r#type).into(), name.into(), transform_stmt_b(root, sb)))
                            .collect(),
                        fallback: fallback.map(|(s, sb)| (s.into(), transform_stmt_b(root, sb)))
                    },
                ast::BlockExpressionKind::Unhandle { code } =>
                    lcr::SBlockTag::Unhandle {
                        what: transform_stmt_b(root, code)
                    },
                ast::BlockExpressionKind::Loop { code } => {
                    let label: Box<str> = rand::distr::Alphabetic.sample_string(&mut rand::rng(), 32).into();  // fixme come up with a REALLY better idea, perhaps level-based labels?
                    
                    lcr::SBlockTag::Block {
                        block: lcr::SBlock {
                            label: label.clone().into(),
                            tag: Box::new(lcr::SBlockTag::Simple {
                                closed: false,
                                decls: vec![],
                                code: vec![
                                    lcr::Instruction::DoBlock(transform_stmt_b(root, code)),
                                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { label }),
                                ],
                            })
                        }
                    }
                }
                ast::BlockExpressionKind::While { check, code, do_first, inverted } => {
                    let label: Box<str> = rand::distr::Alphabetic.sample_string(&mut rand::rng(), 32).into();  // fixme come up with a REALLY better idea, perhaps level-based labels?
                    
                    lcr::SBlockTag::Block {
                        block: lcr::SBlock {
                            label: Some(label.clone()),
                            tag: Box::new(lcr::SBlockTag::Simple {
                                closed: false,
                                decls: vec![],
                                code:
                                    if !do_first {
                                        vec![
                                            lcr::Instruction::DoBlock(lcr::SBlock {
                                                label: None,
                                                tag: Box::new(lcr::SBlockTag::Selector {
                                                    of: transform_expr(root, check),
                                                    cases: vec![(
                                                            lcr::SBlock {
                                                                label: None,
                                                                tag: Box::new(lcr::SBlockTag::Simple {
                                                                    closed: false,
                                                                    decls: vec![],
                                                                    code: vec![lcr::Instruction::LoadLiteral(ast::LiteralExpression::Bool(inverted).into())],
                                                                }),
                                                            },
                                                            lcr::SBlock {
                                                                label: None,
                                                                tag: Box::new(lcr::SBlockTag::Simple {
                                                                    closed: false,
                                                                    decls: vec![],
                                                                    code: vec![lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { value: None, label: label.clone() })],
                                                                }),
                                                            }
                                                        )],
                                                    fallback: None,
                                                })
                                            }),
                                            lcr::Instruction::DoBlock(transform_stmt_b(root, code)),
                                            lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { label })
                                        ]
                                    } else {
                                        vec![
                                            lcr::Instruction::DoBlock(transform_stmt_b(root, code)),
                                            lcr::Instruction::DoBlock(lcr::SBlock {
                                                label: None,
                                                tag: Box::new(lcr::SBlockTag::Selector {
                                                    of: transform_expr(root, check),
                                                    cases: vec![(
                                                        lcr::SBlock {
                                                            label: None,
                                                            tag: Box::new(lcr::SBlockTag::Simple {
                                                                closed: false,
                                                                decls: vec![],
                                                                code: vec![lcr::Instruction::LoadLiteral(ast::LiteralExpression::Bool(!inverted).into())],
                                                            }),
                                                        },
                                                        lcr::SBlock {
                                                            label: None,
                                                            tag: Box::new(lcr::SBlockTag::Simple {
                                                                closed: false,
                                                                decls: vec![],
                                                                code: vec![lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { label })],
                                                            }),
                                                        }
                                                    )],
                                                    fallback: None,
                                                })
                                            }),
                                        ]
                                    }
                            })
                        }
                    }
                },
                ast::BlockExpressionKind::Over { .. } => todo!("protos are not implemented yet, so over blocks are not supported"),
            };

            lcr::SBlock {
                label: block_e.label.map(|s| s.into()),
                tag: Box::new(tag),
            }
        },
        ast::Expression::Construct(cnstr_e) =>
            lcr::SBlock {
                label: None,
                tag: Box::new(lcr::SBlockTag::Simple {
                    closed: false,
                    decls: vec![],
                    code: vec![lcr::Instruction::Construct(
                        match cnstr_e {
                            ast::ConstructExpression::Array { vals } =>
                                lcr::ConstructInstruction::Array {
                                    vals: vals.into_iter()
                                        .map(|e| transform_expr(root, e))
                                        .collect()
                                },
                            ast::ConstructExpression::Data { what, fields } =>
                                lcr::ConstructInstruction::Data {
                                    what: transform_item(root, what).into(),
                                    fields: fields.into_iter()
                                        .map(|(n, e)| (n.into(), transform_expr(root, e)))
                                        .collect(),
                                },
                        },
                    )],
                }),
            },
        ast::Expression::Literal(lit_e) =>
            lcr::SBlock {
                label: None,
                tag: Box::new(lcr::SBlockTag::Simple {
                    closed: false,
                    decls: vec![],
                    code: vec![lcr::Instruction::LoadLiteral(lit_e.into())],
                }),
            },
    }
}

fn transform_item(root: &lcr::Path, item: ast::Item) -> lcr::Path {
    match item.root {
        None => root.clone().combine(item.path),
        Some(ast::ItemRoot::CrateRoot) => item.path[..].into(),
        Some(ast::ItemRoot::LibRoot { lib }) => {
            let mut p: lcr::Path = item.path[..].into();
            p.crate_ = Some(lib.into());
            p
        },
        Some(ast::ItemRoot::ModRoot { updepth }) => root.clone().pop(updepth).combine(item.path),
    }
}

fn transform_im_item(root: &lcr::Path, item: ast::Item) -> lcr::IntermediatePath {
    lcr::IntermediatePath {
        var: (item.root.is_none() && item.path.len() == 1)
            .then_some((**item.path.first().unwrap()).into()),
        path: transform_item(root, item),
    }
}

fn trasform_concrete_type(root: &lcr::Path, ct: ast::ProtocolType) -> lcr::ConcreteTypeRef {
    lcr::ConcreteTypeRef {
        id: transform_item(root, ct.base).into(),
        generics: ct.generics.into_iter().map(|t| transform_type(root, t)).collect()
    }
}

fn transform_type(root: &lcr::Path, t: ast::Type) -> lcr::TypeRef {
    match t {
        ast::Type::Primitive(p) => lcr::TypeRef::Primitive(p.into()),
        ast::Type::Data(i) => lcr::TypeRef::Data(transform_item(root, i).into()),
        ast::Type::Array(t) => lcr::TypeRef::Array(Some(Box::new(transform_type(root, *t)))),
        ast::Type::Op(ps) => lcr::TypeRef::Op(ps.into_iter().map(|ct| trasform_concrete_type(root, ct)).collect()),
        ast::Type::Never => lcr::TypeRef::Never,
    }
}

fn transform_asm_b(root: &lcr::Path, asm_b: ast::AsmBlock) -> lcr::AsmBlock {
    lcr::AsmBlock {
        code: asm_b.instrs.into_iter().map(|instr|
        lcr::AsmInstruction {
            label: instr.label.map(|s| s.into()),
            op: match instr.op {
                ast::AsmOp::Pack { r#type } =>
                    lcr::AsmOp::Pack {
                        r#type: r#type.map_either(
                            |(id, n)| (id.into(), n),
                            |i| transform_item(root, i).into(),
                        ),
                    },
                ast::AsmOp::LoadConstItem { item } =>
                    lcr::AsmOp::LoadConstItem {
                        item: item.map_either(
                            |id| id.into(),
                            |i| i.into(),
                        ),
                    },
                ast::AsmOp::LoadFunction { func } =>
                    lcr::AsmOp::LoadFunction {
                        func: func.map_either(
                            |id| id.into(),
                            |i| transform_item(root, i),
                        ),
                    },
                ast::AsmOp::LoadImplementation { of } =>
                    lcr::AsmOp::LoadImplementation {
                        of: of.map_either(
                            |(id, i)| (id.into(), i),
                            |i| transform_item(root, i),
                        ),
                    },
                ast::AsmOp::LoadSystemItem { id } =>
                    lcr::AsmOp::LoadSystemItem {
                        id: id.map_either(
                            |id| id.into(),
                            |name| name.into(),
                        ),
                    },
                ast::AsmOp::Access { id } =>
                    lcr::AsmOp::Access {
                        id: id.map_right(|f| transform_item(root, f)),
                    },
                ast::AsmOp::Call { which } => lcr::AsmOp::Call { which },
                ast::AsmOp::SystemCall { id } =>
                    lcr::AsmOp::SystemCall {
                        id: id.map_either(
                            |id| id.into(),
                            |name| name.into(),
                        ),
                    },
                ast::AsmOp::Return => lcr::AsmOp::Return,
                ast::AsmOp::Swap { with } => lcr::AsmOp::Swap { with },
                ast::AsmOp::Pull { which } => lcr::AsmOp::Pull { which },
                ast::AsmOp::Pop { count, offset } => lcr::AsmOp::Pop { count, offset },
                ast::AsmOp::Copy { count, offset } => lcr::AsmOp::Copy { count, offset },
                ast::AsmOp::Jump { to, check } =>
                    lcr::AsmOp::Jump {
                        to: to.map_right(|label| label.into()),
                        check,
                    },
            },
        }
        ).collect(),
    }
}
