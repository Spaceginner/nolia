use std::collections::HashMap;
use std::num::NonZero;
use either::Either;
use crate::parser::ast;
use crate::vm;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Path {
    pub crate_: Option<Box<str>>,
    pub parts: Vec<Box<str>>,
}

#[macro_export]
macro_rules! path {
    () => {
        Path { crate_: None, parts: vec![] }
    };
    ($part:ident $(:: $parts:ident)*) => {
        Path { crate_: None, parts: vec![stringify!($part).into(), $(stringify!($parts).into()),*] }
    };
    ($crate_:ident @ $part:ident $(:: $parts:ident)*) => {
        Path { crate_: Some(stringify!($crate_).into()), parts: vec![stringify!($part).into(), $(stringify!($parts).into()),*] }
    };
}


impl<'p, 's> From<&'p [&'s str]> for Path {
    fn from(path: &'p [&'s str]) -> Self {
        Self { parts: path.iter().map(|&p| p.into()).collect(), crate_: None }
    }
}


impl Path {
    pub fn extend(mut self, part: &str) -> Self {
        self.parts.push(part.into());
        self
    }

    pub fn combine(self, path: Vec<&str>) -> Self {
        path.into_iter().fold(self, |p, i| p.extend(i))
    }

    pub fn pop(mut self, n: usize) -> Self {
        (0..n).for_each(|_| { let _ = self.parts.pop().expect("too bad; path breaks out"); });
        self
    }

    pub fn noc(mut self) -> Self {
        self.crate_ = None;
        self
    }
}

// #[derive(Debug, Clone)]
// struct Project {
//     crates: HashMap<Box<str>, Crate>,
// }

#[derive(Debug, Clone, Eq, PartialEq)]
enum IntegerTypeSize {
    Byte,   // 8
    Word,   // 16
    DWord,  // 32
    QWord,  // 64
    OWord,  // 128
}

impl From<ast::IntegerTypeSize> for IntegerTypeSize {
    fn from(value: ast::IntegerTypeSize) -> Self {
        match value {
            ast::IntegerTypeSize::Byte => Self::Byte,
            ast::IntegerTypeSize::Word => Self::Word,
            ast::IntegerTypeSize::DWord => Self::DWord,
            ast::IntegerTypeSize::QWord => Self::QWord,
            ast::IntegerTypeSize::OWord => Self::OWord,
        }
    }
}


#[derive(Debug, Clone, Eq, PartialEq)]
enum PrimitiveType {
    String,
    Char,
    Bool,
    Integer {
        signed: bool,
        size: IntegerTypeSize,
    },
    Float,
    Void,
}


impl From<ast::PrimitiveType> for PrimitiveType {
    fn from(value: ast::PrimitiveType) -> Self {
        match value {
            ast::PrimitiveType::String => Self::String,
            ast::PrimitiveType::Char => Self::Char,
            ast::PrimitiveType::Bool => Self::Bool,
            ast::PrimitiveType::Integer { signed, size } => Self::Integer { signed, size: size.into() },
            ast::PrimitiveType::Void => Self::Void,
        }
    }
}

#[derive(Debug, Clone)]
struct DataType {
    pub fields: Vec<(Box<str>, TypeRef)>
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ConcreteTypeId {
    pub path: Path,
}


impl From<Path> for ConcreteTypeId {
    fn from(path: Path) -> Self {
        Self { path }
    }
}


#[derive(Debug, Clone)]
struct ConcreteTypeRef {
    pub id: ConcreteTypeId,
    pub generics: Vec<TypeRef>,
}


impl From<ConcreteTypeId> for ConcreteTypeRef {
    fn from(id: ConcreteTypeId) -> Self {
        Self { id, generics: vec![] }
    }
}


#[derive(Debug, Clone)]
enum TypeRef {
    Op(Vec<ConcreteTypeRef>),
    Data(ConcreteTypeId),
    Primitive(PrimitiveType),
    Array(Option<Box<TypeRef>>),
    Function(Box<FunctionType>),
    Nothing,
    Never,
}


impl TypeRef {
    pub fn within(&self, bounds: &TypeRef) -> bool {
        match self {
            Self::Op(ops) => todo!("protos are not supported"),
            Self::Data(data) => matches!(bounds, Self::Data(other) if data == other),
            Self::Primitive(lit) => matches!(bounds, Self::Primitive(other) if lit == other),
            Self::Array(None) => true,
            Self::Array(Some(item)) => matches!(bounds, Self::Array(other) if item.within(&*other.as_ref().expect("FIXME needs considiration"))),  // fixme
            Self::Function(_) => false,  // fixme
            Self::Nothing => matches!(bounds, Self::Nothing),
            Self::Never => true,
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionType {
    pub captures: Vec<TypeRef>,
    pub r#return: TypeRef,
    pub generics: GenericDefs,
    pub errors: Vec<ConcreteTypeId>,
}

type GenericDefs = Vec<(Box<str>, Option<TypeRef>)>;

#[derive(Debug, Clone)]
struct ProtocolType {
    pub generics: GenericDefs,
    pub extends: Vec<ConcreteTypeId>,
    pub sigs: Vec<(Box<str>, FunctionType)>
}

#[derive(Debug, Clone)]
struct Declaration {
    name: Box<str>,
    r#type: TypeRef,
    default: Option<SBlock>,
}

#[derive(Debug, Clone)]
struct IntermediatePath {
    var: Option<Box<str>>,
    path: Path,
}

#[derive(Debug, Clone)]
enum ActionInstruction {
    Call {
        what: SBlock,
        args: Vec<SBlock>,
    },
    MethodCall {
        what: SBlock,
        method: Box<str>,
        args: Vec<SBlock>,
    },
    Access {
        of: SBlock,
        field: Box<str>,
    },
    Load {
        item: IntermediatePath,
    },
}

#[derive(Debug, Clone)]
enum LiteralInteger {
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
}


impl From<ast::LiteralInteger> for LiteralInteger {
    fn from(value: ast::LiteralInteger) -> Self {
        match value {
            ast::LiteralInteger::I16(n) => Self::I16(n),
            ast::LiteralInteger::I32(n) => Self::I32(n),
            ast::LiteralInteger::I64(n) => Self::I64(n),
            ast::LiteralInteger::I128(n) => Self::I128(n),
            ast::LiteralInteger::U16(n) => Self::U16(n),
            ast::LiteralInteger::U32(n) => Self::U32(n),
            ast::LiteralInteger::U64(n) => Self::U64(n),
            ast::LiteralInteger::U128(n) => Self::U128(n),
        }
    }
}


impl From<LiteralInteger> for vm::Integer {
    fn from(int: LiteralInteger) -> Self {
        match int {
            LiteralInteger::I16(n) => vm::Integer::I16(n),
            LiteralInteger::I32(n) => vm::Integer::I32(n),
            LiteralInteger::I64(n) => vm::Integer::I64(n),
            LiteralInteger::I128(n) => vm::Integer::I128(n),
            LiteralInteger::U16(n) => vm::Integer::U16(n),
            LiteralInteger::U32(n) => vm::Integer::U32(n),
            LiteralInteger::U64(n) => vm::Integer::U64(n),
            LiteralInteger::U128(n) => vm::Integer::U128(n),
        }
    }
}


#[derive(Debug, Clone)]
enum LiteralValue {
    Integer(LiteralInteger),
    Float(f64),
    Char(char),
    String(Box<str>),
    Bool(bool),
    Void,
}

impl From<ast::LiteralExpression> for LiteralValue {
    fn from(value: ast::LiteralExpression) -> Self {
        match value {
            ast::LiteralExpression::Integer(n) => Self::Integer(n.into()),
            ast::LiteralExpression::Float(f) => Self::Float(f),
            ast::LiteralExpression::Char(c) => Self::Char(c),
            ast::LiteralExpression::String(s) => Self::String(s.into()),
            ast::LiteralExpression::Bool(b) => Self::Bool(b),
            ast::LiteralExpression::Void => Self::Void,
        }
    }
}

impl From<LiteralValue> for vm::ConstItem {
    fn from(value: LiteralValue) -> Self {
        match value {
            LiteralValue::Integer(n) => vm::ConstItem::Integer(n.into()),
            LiteralValue::Float(f) => vm::ConstItem::Float(f),
            LiteralValue::Char(c) => vm::ConstItem::Char(c),
            LiteralValue::String(s) => vm::ConstItem::String(s.to_string()),
            _ => unreachable!("not supported ({value:?})"),
        }
    }
}

#[derive(Debug, Clone)]
enum ConstructInstruction {
    Array {
        vals: Vec<SBlock>,
    },
    Data {
        what: ConcreteTypeId,
        fields: Vec<(Box<str>, SBlock)>
    },
}

#[derive(Debug, Clone)]
enum StatementInstruction {
    Assignment {
        what: SBlock,
        to: SBlock,
    },
    Escape {
        value: Option<SBlock>,
        label: Option<Box<str>>,
    },
    Repeat {
        label: Option<Box<str>>,
    },
    Return {
        value: Option<SBlock>,
    },
    Throw {
        error: SBlock,
    },
}

#[derive(Debug, Clone)]
enum Instruction {
    DoBlock(SBlock),
    DoAction(ActionInstruction),
    LoadLiteral(LiteralValue),
    Construct(ConstructInstruction),
    DoStatement(StatementInstruction),
}


#[derive(Debug, Clone)]
enum SBlockTag {
    Simple {
        decls: Vec<Declaration>,
        code: Vec<Instruction>,
        closed: bool,
    },
    Condition {
        check: SBlock,
        code: SBlock,
        otherwise: Option<SBlock>,
    },
    Selector {
        of: SBlock,
        cases: Vec<(SBlock, SBlock)>,
        fallback: Option<SBlock>,
    },
    Handle {
        what: SBlock,
        handlers: Vec<(ConcreteTypeId, Box<str>, SBlock)>,
        fallback: Option<(Box<str>, SBlock)>,
    },
    Unhandle {
        what: SBlock,
    },
    Loop {
        code: SBlock,
    },
    While {
        code: SBlock,
        check: SBlock,
        do_first: bool,
    },
    Over {
        code: SBlock,
        what: SBlock,
        with: Box<str>,
    }
}

#[derive(Debug, Clone)]
struct SBlock {
    tag: Box<SBlockTag>,
    label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
struct AsmId {
    pub space: Option<NonZero<u32>>,
    pub item: u32,
}


impl From<ast::AsmId> for AsmId {
    fn from(ast::AsmId { space, item }: ast::AsmId) -> Self {
        Self { item, space: NonZero::new(space) }
    }
}


impl From<AsmId> for vm::Id {
    fn from(AsmId { space, item }: AsmId) -> Self {
        Self { space, item }
    }
}


impl From<usize> for vm::Id {
    fn from(item: usize) -> Self {
        Self { space: None, item: item.try_into().unwrap() }
    }
}



#[derive(Debug, Clone)]
enum AsmOp {
    Pack { r#type: Either<ConcreteTypeId, (AsmId, usize)> },
    LoadConstItem { item: Either<LiteralValue, AsmId> },
    LoadFunction { func: Either<Path, AsmId> },
    LoadImplementation { of: Either<Path, (AsmId, u32)> },
    LoadSystemItem { id: Either<Box<str>, AsmId> },
    Access { id: Either<Path, u32> },
    GetType,
    Call { which: usize },
    SystemCall { id: Either<Box<str>, AsmId> },
    Return,
    Swap { with: usize },
    Pull { which: usize },
    Pop { count: usize, offset: usize },
    Copy { which: usize },
    Jump { to: Either<Box<str>, usize>, check: Option<bool> },
}

#[derive(Debug, Clone)]
struct AsmInstruction {
    pub op: AsmOp,
    pub label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
struct AsmBlock {
    pub code: Vec<AsmInstruction>,
}

#[derive(Debug, Clone)]
enum Block {
    Structured(SBlock),
    Asm(AsmBlock),
}

#[derive(Debug, Clone)]
struct Function {
    r#type: FunctionType,
    code: Block,
}


#[derive(Debug, Clone)]
struct Crate {
    deps: Vec<(Box<str>, (u16, u16, u16))>,
    implementation_store: HashMap<ConcreteTypeId, Vec<(ConcreteTypeId, Vec<Function>)>>,  // data type id -> (protocol id, impl funcs)
    function_store: HashMap<Path, Function>,
    method_store: HashMap<TypeRef, Vec<Function>>,
    item_store: HashMap<Path, LiteralValue>,
}


#[derive(Debug, Clone, Default)]
pub struct Compiler {
    crates: HashMap<(Box<str>, (u16, u16, u16)), Crate>,
}


pub struct CrateSource<'s> {
    pub id: (Box<str>, (u16, u16, u16)),
    pub mods: Vec<(Path, ast::Module<'s>)>,
    pub deps: Vec<(Box<str>, (u16, u16, u16))>,
}


#[derive(Debug)]
pub enum CompileError {
    
}


impl Compiler {
    pub fn load_crate<'s>(&mut self, crate_: CrateSource<'s>) {
        let mut impls = HashMap::new();
        let mut funcs = HashMap::new();
        let mut meths = HashMap::new();
        let mut items = HashMap::new();

        for (p, r#mod) in crate_.mods {
            let transform_item = |item: ast::Item| {
                match item.root {
                    None => p.clone().combine(item.path),
                    Some(ast::ItemRoot::CrateRoot) => item.path[..].into(),
                    Some(ast::ItemRoot::LibRoot { lib }) => {
                        let mut p: Path = item.path[..].into();
                        p.crate_ = Some(lib.into());
                        p
                    },
                    Some(ast::ItemRoot::ModRoot { updepth }) => p.clone().pop(updepth).combine(item.path),
                }
            };
            let transform_im_item = |item: ast::Item| {
                IntermediatePath {
                    var: (item.root.is_none() && item.path.len() == 1)
                        .then_some((**item.path.first().unwrap()).into()),
                    path: transform_item(item),
                }
            };
            fn trasform_concrete_type_inner(ct: ast::ProtocolType, ti: &impl Fn(ast::Item) -> Path) -> ConcreteTypeRef {
                ConcreteTypeRef {
                    id: ti(ct.base).into(),
                    generics: ct.generics.into_iter().map(|t| transform_type_inner(t, ti)).collect()
                }
            }
            fn transform_type_inner(t: ast::Type, ti: &impl Fn(ast::Item) -> Path) -> TypeRef {
                match t {
                    ast::Type::Primitive(p) => TypeRef::Primitive(p.into()),
                    ast::Type::Data(i) => TypeRef::Data(ti(i).into()),
                    ast::Type::Array(t) => TypeRef::Array(Some(Box::new(transform_type_inner(*t, ti)))),
                    ast::Type::Op(ps) => TypeRef::Op(ps.into_iter().map(|ct| trasform_concrete_type_inner(ct, ti)).collect()),
                    ast::Type::Never => TypeRef::Never,
                }
            }
            // let transform_concrete_type = |ct| trasform_concrete_type_inner(ct, &transform_item);
            let transform_type = |t| transform_type_inner(t, &transform_item);

            for decl in r#mod.decls {
                match decl {
                    ast::Declaration::Include { item } => {
                        todo!("includes not supported");
                    },
                    ast::Declaration::Data { .. } => { todo!("data not supported"); },
                    ast::Declaration::Protocol { .. } => { todo!("protos not supported"); },
                    ast::Declaration::Implementation { .. } => { todo!("impls not supported"); },
                    ast::Declaration::Function { of, r#impl, name, closure } => {
                        assert!(r#impl.is_none(), "impls not supported");
                        assert!(of.is_none(), "methods not supported");

                        eprintln!("FIXME WARNING - CAPTURES IN FUNCTION SIGNATURES IGNORED (MAKE SYNTAX REPR BETTER)");

                        let f_type = FunctionType {
                            captures: closure.sig.captures.into_iter().map(|c| transform_type(c.r#type)).collect(),
                            r#return: closure.sig.r#return.map(transform_type).unwrap_or(TypeRef::Nothing),
                            generics: closure.sig.generics.defs.into_iter().map(|def| (def.name.into(), def.constraint.map(transform_type))).collect(),
                            errors: closure.sig.errors.into_iter().map(|i| transform_item(i).into()).collect(),
                        };

                        let code = match closure.code {
                            Either::Left(stmt_b) =>
                                Block::Structured({
                                    fn transform_stmt_b<'s>(
                                        stmt_b: ast::StatementBlock<'s>,
                                        tt: &impl Fn(ast::Type<'s>) -> TypeRef,
                                        timi: &impl Fn(ast::Item<'s>) -> IntermediatePath,
                                        ti: &impl Fn(ast::Item<'s>) -> Path,
                                    ) -> SBlock {
                                        let mut decls = Vec::new();
                                        let code = stmt_b.code.into_iter()
                                            .filter_map(|stmt| {
                                                Some(match stmt {
                                                    ast::Statement::Declaration { what, with } => {
                                                        decls.push(Declaration {
                                                            name: what.name.into(),
                                                            r#type: tt(what.r#type),
                                                            default: with.map(|e| transform_expr(e, tt, timi, ti)),
                                                        });
                                                        None?
                                                    },
                                                    ast::Statement::Eval { expr } =>
                                                        Instruction::DoBlock(transform_expr(expr, tt, timi, ti)),
                                                    ast::Statement::Assignment { what, to } =>
                                                        Instruction::DoStatement(StatementInstruction::Assignment {
                                                            what: transform_expr(what, tt, timi, ti),
                                                            to: transform_expr(to, tt, timi, ti),
                                                        }),
                                                    ast::Statement::Escape { value, label } =>
                                                        Instruction::DoStatement(StatementInstruction::Escape {
                                                            value: value.map(|e| transform_expr(e, tt, timi, ti)),
                                                            label: label.map(|s| s.into()),
                                                        }),
                                                    ast::Statement::Repeat { label } =>
                                                        Instruction::DoStatement(StatementInstruction::Repeat {
                                                            label: label.map(|s| s.into()),
                                                        }),
                                                    ast::Statement::Return { value } =>
                                                        Instruction::DoStatement(StatementInstruction::Return {
                                                            value: value.map(|e| transform_expr(e, tt, timi, ti)),
                                                        }),
                                                    ast::Statement::Throw { error } =>
                                                        Instruction::DoStatement(StatementInstruction::Throw {
                                                            error: transform_expr(error, tt, timi, ti)
                                                        }),
                                                })
                                            }).collect();

                                        SBlock {
                                            label: None,
                                            tag: Box::new(SBlockTag::Simple {
                                                code, decls, closed: stmt_b.closed
                                            }),
                                        }
                                    }

                                    fn transform_expr<'s>(
                                        expr: ast::Expression<'s>,
                                        tt: &impl Fn(ast::Type<'s>) -> TypeRef,
                                        timi: &impl Fn(ast::Item<'s>) -> IntermediatePath,
                                        ti: &impl Fn(ast::Item<'s>) -> Path,
                                    ) -> SBlock {
                                        match expr {
                                            ast::Expression::Action(act_e) =>
                                                SBlock {
                                                    tag: Box::new(SBlockTag::Simple {
                                                        closed: false,
                                                        decls: vec![],
                                                        code: vec![Instruction::DoAction(match *act_e {
                                                            ast::ActionExpression::Call { what, args } =>
                                                                ActionInstruction::Call {
                                                                    what: transform_expr(what, tt, timi, ti),
                                                                    args: args.into_iter().map(|e| transform_expr(e, tt, timi, ti)).collect(),
                                                                },
                                                            ast::ActionExpression::MethodCall { what, method, args } =>
                                                                ActionInstruction::MethodCall {
                                                                    what: transform_expr(what, tt, timi, ti),
                                                                    method: method.into(),
                                                                    args: args.into_iter().map(|e| transform_expr(e, tt, timi, ti)).collect()
                                                                },
                                                            ast::ActionExpression::Access { of, field } =>
                                                                ActionInstruction::Access {
                                                                    of: transform_expr(of, tt, timi, ti),
                                                                    field: field.into(),
                                                                },
                                                            ast::ActionExpression::Load { item } =>
                                                                ActionInstruction::Load {
                                                                    item: timi(item),
                                                                },
                                                        })],
                                                    }),
                                                    label: None,
                                                },
                                            ast::Expression::Block(block_e) => {
                                                let tag = match *block_e.kind {
                                                    ast::BlockExpressionKind::Simple { code } => *transform_stmt_b(code, tt, timi, ti).tag,
                                                    ast::BlockExpressionKind::Condition { code, check, otherwise } =>
                                                        SBlockTag::Condition {
                                                            code: transform_stmt_b(code, tt, timi, ti),
                                                            check: transform_expr(check, tt, timi, ti),
                                                            otherwise: otherwise.map(|sb| transform_stmt_b(sb, tt, timi, ti)),
                                                        },
                                                    ast::BlockExpressionKind::Selector { of, fallback, cases } =>
                                                        SBlockTag::Selector {
                                                            of: transform_expr(of, tt, timi, ti),
                                                            cases: cases.into_iter()
                                                                .map(|(e, sb)| (transform_expr(e, tt, timi, ti), transform_stmt_b(sb, tt, timi, ti)))
                                                                .collect(),
                                                            fallback: fallback.map(|sb| transform_stmt_b(sb, tt, timi, ti)),
                                                        },
                                                    ast::BlockExpressionKind::Handle { of, handlers, fallback } =>
                                                        SBlockTag::Handle {
                                                            what: transform_expr(of, tt, timi, ti),
                                                            handlers: handlers.into_iter()
                                                                .map(|(r#type, name, sb)|
                                                                    (ti(r#type).into(), name.into(), transform_stmt_b(sb, tt, timi, ti)))
                                                                .collect(),
                                                            fallback: fallback.map(|(s, sb)| (s.into(), transform_stmt_b(sb, tt, timi, ti)))
                                                        },
                                                    ast::BlockExpressionKind::Unhandle { code } =>
                                                        SBlockTag::Unhandle {
                                                            what: transform_stmt_b(code, tt, timi, ti)
                                                        },
                                                    ast::BlockExpressionKind::Loop { code } =>
                                                        SBlockTag::Loop {
                                                            code: transform_stmt_b(code, tt, timi, ti),
                                                        },
                                                    ast::BlockExpressionKind::While { check, code, do_first } =>
                                                        SBlockTag::While {
                                                            code: transform_stmt_b(code, tt, timi, ti),
                                                            check: transform_expr(check, tt, timi, ti),
                                                            do_first,
                                                        },
                                                    ast::BlockExpressionKind::Over { code, what, with } =>
                                                        SBlockTag::Over {
                                                            code: transform_stmt_b(code, tt, timi, ti),
                                                            what: transform_expr(what, tt, timi, ti),
                                                            with: with.into(),
                                                        },
                                                };

                                                SBlock {
                                                    label: block_e.label.map(|s| s.into()),
                                                    tag: Box::new(tag),
                                                }
                                            },
                                            ast::Expression::Construct(cnstr_e) =>
                                                SBlock {
                                                    label: None,
                                                    tag: Box::new(SBlockTag::Simple {
                                                        closed: false,
                                                        decls: vec![],
                                                        code: vec![Instruction::Construct(
                                                            match cnstr_e {
                                                                ast::ConstructExpression::Array { vals } =>
                                                                    ConstructInstruction::Array {
                                                                        vals: vals.into_iter()
                                                                            .map(|e| transform_expr(e, tt, timi, ti))
                                                                            .collect()
                                                                    },
                                                                ast::ConstructExpression::Data { what, fields } =>
                                                                    ConstructInstruction::Data {
                                                                        what: ti(what).into(),
                                                                        fields: fields.into_iter()
                                                                            .map(|(n, e)| (n.into(), transform_expr(e, tt, timi, ti)))
                                                                            .collect(),
                                                                    },
                                                            },
                                                        )],
                                                    }),
                                                },
                                            ast::Expression::Literal(lit_e) =>
                                                SBlock {
                                                    label: None,
                                                    tag: Box::new(SBlockTag::Simple {
                                                        closed: false,
                                                        decls: vec![],
                                                        code: vec![Instruction::LoadLiteral(lit_e.into())],
                                                    }),
                                                },
                                        }
                                    }

                                    transform_stmt_b(stmt_b, &transform_type, &transform_im_item, &transform_item)
                                }),
                            Either::Right(asm) => {
                                Block::Asm(AsmBlock {
                                    code: asm.instrs.into_iter().map(
                                        |instr| AsmInstruction {
                                            label: instr.label.map(|s| s.into()),
                                            op: match instr.op {
                                                ast::AsmOp::Pack { r#type } =>
                                                    AsmOp::Pack {
                                                        r#type: r#type.map_either(
                                                            |i| transform_item(i).into(),
                                                            |(id, n)| (id.into(), n),
                                                        ),
                                                    },
                                                ast::AsmOp::LoadConstItem { item } =>
                                                    AsmOp::LoadConstItem {
                                                        item: item.map_either(
                                                            |i| i.into(),
                                                            |id| id.into(),
                                                        ),
                                                    },
                                                ast::AsmOp::LoadFunction { func } =>
                                                    AsmOp::LoadFunction {
                                                        func: func.map_either(
                                                            |i| transform_item(i),
                                                            |id| id.into(),
                                                        ),
                                                    },
                                                ast::AsmOp::LoadImplementation { of } =>
                                                    AsmOp::LoadImplementation {
                                                        of: of.map_either(
                                                            |i| transform_item(i),
                                                            |(id, i)| (id.into(), i),
                                                        ),
                                                    },
                                                ast::AsmOp::LoadSystemItem { id } =>
                                                    AsmOp::LoadSystemItem {
                                                        id: id.map_either(
                                                            |name| name.into(),
                                                            |id| id.into(),
                                                        ),
                                                    },
                                                ast::AsmOp::Access { id } =>
                                                    AsmOp::Access {
                                                        id: id.map_left(|f| transform_item(f)),
                                                    },
                                                ast::AsmOp::GetType => AsmOp::GetType,
                                                ast::AsmOp::Call { which } => AsmOp::Call { which },
                                                ast::AsmOp::SystemCall { id } =>
                                                    AsmOp::SystemCall {
                                                        id: id.map_either(
                                                            |name| name.into(),
                                                            |id| id.into(),
                                                        ),
                                                    },
                                                ast::AsmOp::Return => AsmOp::Return,
                                                ast::AsmOp::Swap { with } => AsmOp::Swap { with },
                                                ast::AsmOp::Pull { which } => AsmOp::Pull { which },
                                                ast::AsmOp::Pop { count, offset } => AsmOp::Pop { count, offset },
                                                ast::AsmOp::Copy { which } => AsmOp::Copy { which },
                                                ast::AsmOp::Jump { to, check } =>
                                                    AsmOp::Jump {
                                                        to: to.map_left(|label| label.into()),
                                                        check,
                                                    },
                                            },
                                        },
                                    ).collect(),
                                })
                            },
                        };

                        funcs.insert(
                            p.clone().extend(name),
                            Function { r#type: f_type, code },
                        );
                    },
                    ast::Declaration::Const { what: ast::Capture { name, .. }, val } => {
                        eprintln!("FIXME WARNING - TYPE IN CONSTS IGNORED");
                        items.insert(
                            p.clone().extend(name),
                            val.into(),
                        );
                    },
                };
            };
        };

        self.crates.insert(
            crate_.id,
            Crate {
                deps: crate_.deps,
                implementation_store: impls,
                method_store: meths,
                function_store: funcs,
                item_store: items,
            },
        );
    }

    // xxx should be refactored back
    fn lit_type(lit: &LiteralValue) -> PrimitiveType {
        match lit {
            LiteralValue::Integer(LiteralInteger::I16(_)) => PrimitiveType::Integer { signed: true, size: IntegerTypeSize::Word },
            LiteralValue::Integer(LiteralInteger::I32(_)) => PrimitiveType::Integer { signed: true, size: IntegerTypeSize::DWord },
            LiteralValue::Integer(LiteralInteger::I64(_)) => PrimitiveType::Integer { signed: true, size: IntegerTypeSize::QWord },
            LiteralValue::Integer(LiteralInteger::I128(_)) => PrimitiveType::Integer { signed: true, size: IntegerTypeSize::OWord },
            LiteralValue::Integer(LiteralInteger::U16(_)) => PrimitiveType::Integer { signed: false, size: IntegerTypeSize::Word },
            LiteralValue::Integer(LiteralInteger::U32(_)) => PrimitiveType::Integer { signed: false, size: IntegerTypeSize::DWord },
            LiteralValue::Integer(LiteralInteger::U64(_)) => PrimitiveType::Integer { signed: false, size: IntegerTypeSize::QWord },
            LiteralValue::Integer(LiteralInteger::U128(_)) => PrimitiveType::Integer { signed: false, size: IntegerTypeSize::OWord },
            LiteralValue::Char(_) => PrimitiveType::Char,
            LiteralValue::String(_) => PrimitiveType::String,
            LiteralValue::Float(_) => PrimitiveType::Float,
            LiteralValue::Bool(_) => PrimitiveType::Bool,
            LiteralValue::Void => PrimitiveType::Void,
        }
    }

    pub fn compile(self, entry_func: Option<Path>) -> Result<(Vec<vm::CrateDeclaration>, Option<(vm::CrateId, u32)>), CompileError> {
        let mut entry_func_id = None;
        Ok((self.crates.into_iter().map(|(crate_id_raw, crate_)| {
            let (mut items, item_map) =
                crate_.item_store.into_iter().enumerate()
                    .map(|(i, (path, item))| {
                        let item_type = Self::lit_type(&item);
                        (item.into(), (path, (i, item_type)))
                    })
                    .collect::<(Vec<_>, HashMap<_, _>)>();

            let func_map = crate_.function_store.iter().enumerate()
                .map(|(i, (p, f))| (p.clone(), (i, f.r#type.clone())))
                .collect::<HashMap<_, _>>();

            let funcs = crate_.function_store.into_values()
                .map(|func| vm::FunctionDeclaration {
                        code: match func.code {
                            Block::Structured(s_block) => {
                                fn resolve_type(
                                    s_block: &SBlock,
                                    func_map: &HashMap<Path, (usize, FunctionType)>,
                                    var_scope: &(HashMap<Box<str>, (usize, TypeRef)>, usize),
                                    item_map: &HashMap<Path, (usize, PrimitiveType)>,
                                    rec: bool,
                                ) -> TypeRef {
                                    match &*s_block.tag {
                                        SBlockTag::Simple { code, closed, decls } => {
                                            assert!(decls.is_empty(), "decls are not supported while type resolving");
                                            if *closed || code.is_empty() {
                                                TypeRef::Nothing
                                            } else {
                                                for instr in code.iter() {
                                                    return match instr {
                                                        Instruction::DoStatement(StatementInstruction::Return { .. })
                                                        | Instruction::DoStatement(StatementInstruction::Repeat { .. })
                                                        | Instruction::DoStatement(StatementInstruction::Throw { .. }) if !rec =>
                                                            TypeRef::Never,
                                                        Instruction::DoAction(ActionInstruction::Call { what, .. }) =>
                                                            if !rec
                                                                && let TypeRef::Function(func) = resolve_type(what, func_map, var_scope, item_map, rec)
                                                                && matches!(func.r#return, TypeRef::Never) {
                                                                TypeRef::Never
                                                            } else {
                                                                continue
                                                            }
                                                        _ => continue,
                                                    };
                                                };

                                                match code.last().unwrap() {
                                                    Instruction::LoadLiteral(lit) =>
                                                        TypeRef::Primitive(Compiler::lit_type(lit)),
                                                    Instruction::Construct(ConstructInstruction::Data { what, .. }) =>
                                                        TypeRef::Data(what.clone()),
                                                    Instruction::Construct(ConstructInstruction::Array { vals, .. }) =>
                                                    // fixme forbid arrays of nothing
                                                        TypeRef::Array(vals.first().map(|b| Box::new(resolve_type(b, func_map, var_scope, item_map, rec)))),
                                                    Instruction::DoBlock(b) => resolve_type(b, func_map, var_scope, item_map, true),
                                                    Instruction::DoAction(ActionInstruction::Load { item }) => {
                                                        if let Some(var_name) = item.var.as_ref()
                                                            && let Some((_, r#type)) = var_scope.0.get(var_name) {
                                                            r#type.clone()
                                                        } else {
                                                            assert!(item.path.crate_.is_none());
                                                            item_map.get(&item.path)
                                                                .map(|(_, t)| TypeRef::Primitive(t.clone()))
                                                                .unwrap_or_else(|| {
                                                                    TypeRef::Function(Box::new(func_map.get(&item.path).unwrap().1.clone()))
                                                                })
                                                        }
                                                    },
                                                    Instruction::DoAction(ActionInstruction::Access { .. }) => todo!("fields are not supported"),
                                                    Instruction::DoAction(ActionInstruction::MethodCall { .. }) => todo!("methods are not supported"),
                                                    Instruction::DoAction(ActionInstruction::Call { what, .. }) =>
                                                        if let TypeRef::Function(func) = resolve_type(what, func_map, var_scope, item_map, rec) {
                                                            func.r#return
                                                        } else {
                                                            panic!("cant call non functions")
                                                        },
                                                    Instruction::DoStatement(StatementInstruction::Assignment { .. }) => TypeRef::Nothing,
                                                    _ => unreachable!()
                                                }
                                            }
                                        },
                                        SBlockTag::Handle { what, .. }
                                        | SBlockTag::Unhandle { what } =>
                                            resolve_type(what, func_map, var_scope, item_map, rec),
                                        SBlockTag::Condition { code, .. } =>
                                            resolve_type(code, func_map, var_scope, item_map, true),
                                        SBlockTag::Selector { cases, .. } =>
                                            cases.first().map(|(_, b)| resolve_type(b, func_map, var_scope, item_map, true)).unwrap_or(TypeRef::Nothing),
                                        SBlockTag::Loop { code } =>
                                            if let TypeRef::Never = resolve_type(code, func_map, var_scope, item_map, rec) {
                                                TypeRef::Never
                                            } else {
                                                if let SBlockTag::Simple { code, .. } = &*code.tag {
                                                    for instr in code {
                                                        match instr {
                                                            Instruction::DoStatement(StatementInstruction::Escape { value: Some(value), .. }) =>
                                                                return resolve_type(value, func_map, var_scope, item_map, rec),
                                                            _ => {},
                                                        };
                                                    };
                                                    TypeRef::Never
                                                } else {
                                                    panic!("uh oh")
                                                }
                                            },
                                        SBlockTag::While { code, .. }
                                        | SBlockTag::Over { code, .. } =>
                                            TypeRef::Nothing,
                                    }
                                }

                                fn compile_instruction(
                                    instr: Instruction,
                                    block_starts_at: usize,
                                    this_instr_at: usize,
                                    var_scope: &mut (HashMap<Box<str>, (usize, TypeRef)>, usize),
                                    label_scope: &mut HashMap<&str, usize>,
                                    func_map: &HashMap<Path, (usize, FunctionType)>,
                                    item_map: &HashMap<Path, (usize, PrimitiveType)>,
                                    items: &mut Vec<vm::ConstItem>,
                                ) -> Vec<vm::Op> {
                                    match instr {
                                        Instruction::DoBlock(block) =>
                                            compile_s_block(block, this_instr_at, var_scope, label_scope, func_map, item_map, items),
                                        Instruction::LoadLiteral(lit) =>
                                            vec![match lit {
                                                LiteralValue::Bool(bool) => vm::Op::LoadSystemItem { id: (bool as usize).into() },
                                                LiteralValue::Void => vm::Op::LoadSystemItem { id: 2.into() },
                                                lit => vm::Op::LoadConstItem {
                                                    id: {
                                                        let id = items.len();
                                                        items.push(lit.into());
                                                        id.into()
                                                    },
                                                },
                                            }],
                                        Instruction::DoAction(ActionInstruction::Access { of, field }) => todo!("fields access not supported"),
                                        Instruction::DoAction(ActionInstruction::Call { what, args }) => {
                                            if let TypeRef::Function(func) = resolve_type(&what, func_map, var_scope, item_map, false) {
                                                let FunctionType { generics, captures, r#return, .. } = *func;
                                                assert!(generics.is_empty(), "generics are not supported");

                                                if args.len() != captures.len() {
                                                    panic!("function expected different number of arguments");
                                                };

                                                if !args.iter().map(|arg| resolve_type(arg, func_map, var_scope, item_map, false)).zip(captures.into_iter()).all(|(arg, cap)| arg.within(&cap)) {
                                                    panic!("some args have mismatched types");
                                                };

                                                let get_func = compile_s_block(what, this_instr_at, var_scope, label_scope, func_map, item_map, items);
                                                let mut offset = get_func.len();
                                                let mut args_code = Vec::new();
                                                let arg_count = args.len();
                                                for arg in args {
                                                    let mut arg_code = compile_s_block(arg, this_instr_at+offset, var_scope, label_scope, func_map, item_map, items);
                                                    offset += arg_code.len();
                                                    args_code.append(&mut arg_code);
                                                };

                                                [
                                                    get_func,
                                                    args_code,
                                                    vec![
                                                        vm::Op::Call { which: arg_count },
                                                        vm::Op::Pop {
                                                            count: arg_count + 1,
                                                            offset: if let TypeRef::Nothing = r#return { 0 } else { 1 },
                                                        },
                                                    ]
                                                ].concat()
                                            } else {
                                                panic!("cant call not functions")
                                            }
                                        },
                                        Instruction::DoAction(ActionInstruction::MethodCall { what, method, args }) => todo!("methods are not supported"),
                                        Instruction::DoAction(ActionInstruction::Load { item }) =>  {
                                            if let Some(var) = item.var.as_ref()
                                                && let Some(&(i, _)) = var_scope.0.get(var) {
                                                // todo remove clone
                                                let cur_top = var_scope.0.iter().find(|(_, &(i, _))| i == var_scope.1).unwrap().0.clone();
                                                var_scope.0.get_mut(&cur_top).unwrap().0 = i;
                                                vec![vm::Op::Copy { which: var_scope.1-i }]
                                            } else {
                                                assert!(item.path.crate_.is_none());
                                                if let Some((i, _)) = item_map.get(&item.path) {
                                                    vec![vm::Op::LoadConstItem { id: (*i).into() }]
                                                } else if let Some((i, _)) = func_map.get(&item.path) {
                                                    vec![vm::Op::LoadFunction { id: (*i).into() }]
                                                } else {
                                                    panic!("such item doesnt exist");
                                                }
                                            }
                                        },
                                        Instruction::Construct(ConstructInstruction::Data { what, fields }) => todo!("datas are not supported"),
                                        Instruction::Construct(ConstructInstruction::Array { vals }) => todo!("arrays are not supported"),
                                        Instruction::DoStatement(StatementInstruction::Assignment { what, to }) => todo!(),
                                        Instruction::DoStatement(StatementInstruction::Repeat { label }) => todo!(),
                                        Instruction::DoStatement(StatementInstruction::Escape { label, value }) => todo!(),
                                        Instruction::DoStatement(StatementInstruction::Return { value }) => todo!(),
                                        Instruction::DoStatement(StatementInstruction::Throw { error }) => todo!("exceptions are not supported"),
                                    }
                                }

                                fn compile_s_block(
                                    mut s_block: SBlock,
                                    starts_at: usize,
                                    var_scope: &mut (HashMap<Box<str>, (usize, TypeRef)>, usize),
                                    label_scope: &mut HashMap<&str, usize>,
                                    func_map: &HashMap<Path, (usize, FunctionType)>,
                                    item_map: &HashMap<Path, (usize, PrimitiveType)>,
                                    items: &mut Vec<vm::ConstItem>,
                                ) -> Vec<vm::Op> {
                                    let self_type = resolve_type(&s_block, func_map, var_scope, item_map, false);

                                    match *s_block.tag {
                                        SBlockTag::Simple { code, mut decls, closed } => {
                                            let mut code_offset = 0;
                                            let init_code = decls.iter_mut()
                                                .map(|Declaration { name, default, r#type }| {
                                                    if var_scope.0.contains_key(&name[..]) {
                                                        panic!("variable shadowing is forbidden");
                                                    };

                                                    var_scope.0.insert(name.clone(), (var_scope.1+1, r#type.clone()));
                                                    var_scope.1 += 1;

                                                    if let Some(val) = default.take() {
                                                        let def_code = compile_s_block(val, code_offset, var_scope, label_scope, func_map, item_map, items);
                                                        code_offset += def_code.len();
                                                        def_code
                                                    } else {
                                                        code_offset += 1;
                                                        vec![vm::Op::LoadSystemItem { id: 2.into() }]  // fixme soft code this
                                                    }
                                                })
                                                .reduce(|mut a, mut b| { a.append(&mut b); a })
                                                .unwrap_or_else(|| vec![]);

                                            let last_instr_i = code.len() - 1;
                                            let mut offset = init_code.len();
                                            let main_code = code.into_iter().enumerate()
                                                .map(|(i, instr)| {
                                                    let ret_something = !resolve_type(
                                                        &SBlock {
                                                            label: None,
                                                            tag: Box::new(SBlockTag::Simple {
                                                                closed: false,
                                                                decls: vec![],
                                                                code: vec![instr.clone()],
                                                            }),
                                                        },
                                                        func_map,
                                                        var_scope,
                                                        item_map,
                                                        false,
                                                    ).within(&TypeRef::Nothing);
                                                    let mut code = compile_instruction(instr, starts_at, starts_at + offset, var_scope, label_scope, func_map, item_map, items);
                                                    offset += code.len();
                                                    if ret_something && (closed || i != last_instr_i) {
                                                        code.push(vm::Op::Pop { count: 1, offset: 0 });
                                                    };
                                                    code
                                                })
                                                .reduce(|mut a, mut b| { a.append(&mut b); a })
                                                .unwrap_or_else(|| vec![]);

                                            let deinit_code = decls.iter()
                                                .map(|Declaration { name, .. }| {
                                                    let cur_i = var_scope.0[&name[..]].0;
                                                    let cur_top = var_scope.0.iter()
                                                        .find_map(
                                                            |(name, &(pos, _))|
                                                            (pos == var_scope.1).then_some(name.clone())
                                                        ).unwrap();

                                                    let pop_at = var_scope.1 - cur_i;
                                                    var_scope.0.get_mut(&cur_top).unwrap().0 = cur_i;
                                                    var_scope.0.remove(&name[..]);
                                                    var_scope.1 -= 1;

                                                    vm::Op::Pop { count: 1, offset: pop_at }
                                                })
                                                .collect();

                                            [init_code, main_code, deinit_code].concat()
                                        },
                                        SBlockTag::Condition { code, check, otherwise } => {
                                            if let Some(b) = otherwise.as_ref() {
                                                if !resolve_type(b, func_map, var_scope, item_map, false).within(&self_type) {
                                                    panic!("branch is not of the same return type");
                                                };
                                            };

                                            if !resolve_type(&check, func_map, var_scope, item_map, false).within(&TypeRef::Primitive(PrimitiveType::Bool)) {
                                                panic!("check is not bool");
                                            };

                                            let check_code = compile_s_block(check, starts_at, var_scope, label_scope, func_map, item_map, items);

                                            let main_pos = starts_at + check_code.len() + 2;
                                            let main_code = compile_s_block(code, main_pos, var_scope, label_scope, func_map, item_map, items);
                                            let after_main_pos = main_pos + main_code.len() + 1;

                                            let otherwise_code = otherwise.map(|b| compile_s_block(b, after_main_pos, var_scope, label_scope, func_map, item_map, items));
                                            let after_otherwise_pos = after_main_pos + otherwise_code.as_ref().map_or(0, |c| c.len()) + 2;

                                            if let Some(otherwise) = otherwise_code {
                                                [
                                                    check_code,
                                                    vec![
                                                        vm::Op::Jump {
                                                            to: main_pos,
                                                            check: Some(true),
                                                        },
                                                        vm::Op::Jump {
                                                            to: after_main_pos + 1,
                                                            check: None,
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        }
                                                    ],
                                                    main_code,
                                                    vec![
                                                        vm::Op::Jump {
                                                            to: after_otherwise_pos,
                                                            check: None,
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        }
                                                    ],
                                                    otherwise,
                                                ].concat()
                                            } else {
                                                [
                                                    check_code,
                                                    vec![
                                                        // xxx introduce a jump-else kind of instr
                                                        vm::Op::Jump {
                                                            to: main_pos,
                                                            check: Some(true),
                                                        },
                                                        vm::Op::Jump {
                                                            to: after_main_pos,
                                                            check: None,
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        }
                                                    ],
                                                    main_code,
                                                ].concat()
                                            }
                                        },
                                        SBlockTag::Selector { .. } => todo!(),
                                        SBlockTag::Handle { .. } => todo!(),
                                        SBlockTag::Unhandle { .. } => todo!(),
                                        SBlockTag::Loop { code } => {
                                            if !resolve_type(&code, func_map, var_scope, item_map, false).within(&TypeRef::Nothing) {
                                                panic!("code mustn't return anything (only through escaping)");
                                            };

                                            [
                                                compile_s_block(code, starts_at, var_scope, label_scope, func_map, item_map, items),
                                                vec![vm::Op::Jump {
                                                    to: starts_at,
                                                    check: None
                                                }],
                                            ].concat()
                                        },
                                        // todo merge with simple loop
                                        SBlockTag::While { code, check, do_first } => {
                                            if !resolve_type(&code, func_map, var_scope, item_map, false).within(&TypeRef::Nothing) {
                                                panic!("code mustn't return anything");
                                            };

                                            if !resolve_type(&check, func_map, var_scope, item_map, false).within(&TypeRef::Primitive(PrimitiveType::Bool)) {
                                                panic!("check is not bool");
                                            };

                                            if do_first {
                                                let main_pos = starts_at + 2;
                                                let main_code = compile_s_block(code, starts_at, var_scope, label_scope, func_map, item_map, items);
                                                let check_pos = main_pos + main_code.len();
                                                let check_code = compile_s_block(check, check_pos, var_scope, label_scope, func_map, item_map, items);

                                                [
                                                    vec![
                                                        vm::Op::Jump {
                                                            to: main_pos,
                                                            check: None,
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        },
                                                    ],
                                                    main_code,
                                                    check_code,
                                                    vec![
                                                        vm::Op::Jump {
                                                            to: main_pos - 1,
                                                            check: Some(true),
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        },
                                                    ],
                                                ].concat()
                                            } else {
                                                let check_code = compile_s_block(check, starts_at, var_scope, label_scope, func_map, item_map, items);
                                                let main_pos = starts_at + check_code.len() + 2;
                                                let main_code = compile_s_block(code, main_pos, var_scope, label_scope, func_map, item_map, items);
                                                let loop_end_pos = main_pos + main_code.len() + 2;

                                                [
                                                    check_code,
                                                    vec![
                                                        vm::Op::Jump {
                                                            to: main_pos,
                                                            check: Some(true),
                                                        },
                                                        vm::Op::Jump {
                                                            to: loop_end_pos,
                                                            check: None,
                                                        },
                                                        vm::Op::Pop {
                                                            count: 1,
                                                            offset: 0,
                                                        }
                                                    ],
                                                    main_code,
                                                    vec![vm::Op::Jump {
                                                        to: starts_at,
                                                        check: None,
                                                    }],
                                                ].concat()
                                            }
                                        },
                                        SBlockTag::Over { .. } => todo!(),
                                    }
                                }

                                if !resolve_type(&s_block, &func_map, &(HashMap::new(), 0), &item_map, false).within(&func.r#type.r#return) {
                                    panic!("return value is of wrong return type");
                                };

                                let mut code = compile_s_block(
                                    s_block,
                                    0,
                                    &mut (HashMap::new(), 0),
                                    &mut HashMap::new(),
                                    &func_map,
                                    &item_map,
                                    &mut items,
                                );
                                code.push(vm::Op::Return);
                                code
                            },
                            Block::Asm(asm) => {
                                let label_store = asm.code.iter().enumerate()
                                    .filter_map(|(i, instr)| Some((instr.label.clone()?, i)))
                                    .collect::<HashMap<_, _>>();

                                asm.code.into_iter().map(|instr| match instr.op {
                                    AsmOp::Pack { .. } => { todo!("pack instr is not supported") }
                                    AsmOp::LoadConstItem { item } =>
                                        vm::Op::LoadConstItem {
                                            id: match item {
                                                Either::Right(id) => id.into(), // todo add a way to refer to actual const items
                                                Either::Left(val) => {
                                                    let i = items.len();
                                                    items.push(val.into());
                                                    i.into()
                                                },
                                            }
                                        },
                                    AsmOp::LoadFunction { func } =>
                                        vm::Op::LoadFunction {
                                            id: match func {
                                                Either::Right(id) => id.into(),
                                                Either::Left(p) => {
                                                    assert!(p.crate_.is_none());
                                                    func_map[&p].0.into()
                                                },
                                            },
                                        },
                                    AsmOp::LoadImplementation { .. } => { todo!("load impl instr is not supported") },
                                    AsmOp::LoadSystemItem { id } =>
                                        vm::Op::LoadSystemItem {
                                            id: match id {
                                                Either::Right(id) => id.into(),
                                                Either::Left(name) =>
                                                    // fixme soft-code these values (remove magicness)
                                                    match &name[..] {
                                                        "false" => 0,
                                                        "true" => 1,
                                                        "void" => 2,
                                                        _ => panic!("unknown sys item")
                                                    }.into()
                                            },
                                        },
                                    AsmOp::Access { .. } => { todo!("access instr is not supported") }
                                    AsmOp::GetType => vm::Op::GetType,
                                    AsmOp::Call { which } => vm::Op::Call { which },
                                    AsmOp::SystemCall { id } =>
                                        vm::Op::SystemCall {
                                            id: match id {
                                                Either::Right(id) => id.into(),
                                                Either::Left(name) =>
                                                    // fixme soft-code these values (remove magicness)
                                                    match &name[..] {
                                                        "panic" => 0,
                                                        "println" => 1,
                                                        _ => panic!("unknown sys call")
                                                    }.into(),
                                            },
                                        },
                                    AsmOp::Return => vm::Op::Return,
                                    AsmOp::Swap { with } => vm::Op::Swap { with },
                                    AsmOp::Pull { which } => vm::Op::Pull { which },
                                    AsmOp::Pop { count, offset } => vm::Op::Pop { count, offset },
                                    AsmOp::Copy { which } => vm::Op::Copy { which },
                                    AsmOp::Jump { to, check } =>
                                        vm::Op::Jump {
                                            check,
                                            to: match to {
                                                Either::Right(i) => i,
                                                Either::Left(label) => label_store[&label],
                                            },
                                        },
                                }).collect()
                            }
                        }
                    })
                .collect();

            let crate_id = vm::CrateId::new(&crate_id_raw.0, crate_id_raw.1);

            if let Some(path) = entry_func.as_ref() {
                if path.crate_.as_ref().unwrap() == &crate_id_raw.0 {
                    entry_func_id = Some((crate_id, func_map[&path.clone().noc()].0 as u32));
                };
            };

            vm::CrateDeclaration {
                id: crate_id,
                dependencies: Vec::new(),
                items,
                functions: funcs,
                implementations: HashMap::new(),
            }
        }).collect(), entry_func_id))
    }
}
