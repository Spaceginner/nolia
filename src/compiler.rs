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

#[derive(Debug, Clone)]
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


#[derive(Debug, Clone)]
enum PrimitiveType {
    String,
    Char,
    Bool,
    Integer {
        signed: bool,
        size: IntegerTypeSize,
    },
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
enum TypeRef {
    Op(Vec<ConcreteTypeRef>),
    Concrete(ConcreteTypeRef),
    Primitive(PrimitiveType),
    Array(Box<TypeRef>),
    Never,
}

#[derive(Debug, Clone)]
struct FunctionType {
    pub captures: Vec<TypeRef>,
    pub r#return: Option<TypeRef>,
    pub generics: GenericDefs,
    pub errors: Vec<ConcreteTypeRef>,
}

type GenericDefs = Vec<(Box<str>, Option<TypeRef>)>;

#[derive(Debug, Clone)]
struct ProtocolType {
    pub generics: GenericDefs,
    pub extends: Vec<ConcreteTypeId>,
    pub sigs: Vec<(Box<str>, FunctionType)>
}

#[derive(Debug, Clone)]
enum Type {
    Never,
    Primitive(PrimitiveType),
    Data(DataType),
    Protocol(ProtocolType),
    Array(Box<Type>),
}

#[derive(Debug, Clone)]
enum BlockTag {
    Simple,
    Condition {
        check: SBlock,
        otherwise: Option<SBlock>,
    },
    Selector {
        cases: Vec<(SBlock, SBlock)>,
        fallback: Option<SBlock>,
    },
    Handle {
        handlers: Vec<(ConcreteTypeId, Box<str>, SBlock)>,
        fallback: Option<(Box<str>, SBlock)>,
    },
    Unhandle,
    Loop,
    While {
        check: SBlock,
    },
    DoWhile {
        check: SBlock,
    },
    Over {
        what: SBlock,
        with: Box<str>,
    }
}

#[derive(Debug, Clone)]
struct Declaration {
    name: Box<str>,
    r#type: Type,
}

#[derive(Debug, Clone)]
struct IntermediatePath {
    var: Box<str>,
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
        item: Either<IntermediatePath, Path>,
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
            _ => unreachable!("not supported"),
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
struct SBlock {
    tag: Box<BlockTag>,
    label: Option<Box<str>>,
    decls: Vec<Declaration>,
    code: Vec<Instruction>,
    closed: bool,
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
    Call,
    SystemCall { id: Either<Box<str>, AsmId> },
    Return,
    Swap { with: usize },
    Pop { count: usize, offset: usize },
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
    type_store: HashMap<Path, Type>,
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
        let mut types = HashMap::new();
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
                if item.root.is_none() && item.path.len() == 1 {
                    Either::Left(IntermediatePath { var: (**item.path.first().unwrap()).into(), path: transform_item(item) })
                } else {
                    Either::Right(transform_item(item))
                }
            };
            fn trasform_concrete_type_inner(ct: ast::ConcreteType, ti: &impl Fn(ast::Item) -> Path) -> ConcreteTypeRef {
                ConcreteTypeRef {
                    id: ti(ct.item).into(),
                    generics: ct.generics.into_iter().map(|t| transform_type_inner(t, ti)).collect()
                }
            }
            fn transform_type_inner(t: ast::Type, ti: &impl Fn(ast::Item) -> Path) -> TypeRef {
                match t {
                    ast::Type::Primitive(p) => TypeRef::Primitive(p.into()),
                    ast::Type::Concrete(c) => TypeRef::Concrete(trasform_concrete_type_inner(c, ti)),
                    ast::Type::Array(t) => TypeRef::Array(Box::new(transform_type_inner(*t, ti))),
                    ast::Type::Op(ps) => TypeRef::Op(ps.into_iter().map(|ct| trasform_concrete_type_inner(ct, ti)).collect()),
                    ast::Type::Never => TypeRef::Never,
                }
            }
            let transform_concrete_type = |ct| trasform_concrete_type_inner(ct, &transform_item);
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
                            r#return: closure.sig.r#return.map(transform_type),
                            generics: closure.sig.generics.defs.into_iter().map(|def| (def.name.into(), def.constraint.map(transform_type))).collect(),
                            errors: closure.sig.errors.into_iter().map(transform_concrete_type).collect(),
                        };

                        let code = match closure.code {
                            Either::Left(stmt) => todo!("statements arent supported"),
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
                                                ast::AsmOp::Call => AsmOp::Call,
                                                ast::AsmOp::SystemCall { id } =>
                                                    AsmOp::SystemCall {
                                                        id: id.map_either(
                                                            |name| name.into(),
                                                            |id| id.into(),
                                                        ),
                                                    },
                                                ast::AsmOp::Return => AsmOp::Return,
                                                ast::AsmOp::Swap { with } => AsmOp::Swap { with },
                                                ast::AsmOp::Pop { count, offset } => AsmOp::Pop { count, offset },
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
                type_store: types,
                implementation_store: impls,
                method_store: meths,
                function_store: funcs,
                item_store: items,
            },
        );
    }

    pub fn compile(self, entry_func: Option<Path>) -> Result<(Vec<vm::CrateDeclaration>, Option<(vm::CrateId, u32)>), CompileError> {
        let mut entry_func_id = None;
        Ok((self.crates.into_iter().map(|(crate_id_raw, crate_)| {
            let (mut items, item_map) =
                crate_.item_store.into_iter().enumerate()
                    .map(|(i, (path, item))| (item.into(), (path, i)))
                    .collect::<(Vec<_>, HashMap<_, _>)>();

            let func_map = crate_.function_store.iter().enumerate()
                .map(|(i, (p, f))| (p.clone(), (i, f.r#type.clone())))
                .collect::<HashMap<_, _>>();

            let funcs = crate_.function_store.into_values()
                .map(|func| vm::FunctionDeclaration {
                        code: match func.code {
                            Block::Structured(s_block) => todo!("structured blocks are not supported"),
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
                                    AsmOp::Call => vm::Op::Call,
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
                                    AsmOp::Pop { count, offset } => vm::Op::Pop { count, offset },
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
