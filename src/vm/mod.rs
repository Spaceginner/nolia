use std::collections::HashMap;
use std::num::NonZero;
use std::ops::{ControlFlow, Index, Range, RangeTo};
use crate::parser::ast;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id {
    pub space: Option<NonZero<u32>>,
    pub item: u32,
}

impl From<ast::AsmId> for Id {
    fn from(value: ast::AsmId) -> Self {
        Self {
            space: NonZero::new(value.space),
            item: value.item
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub enum Op {
    Pack { id: Id, count: usize },
    LoadConstItem { id: Id },
    LoadFunction { id: Id },
    LoadImplementation { id: Id, func: u32 },
    LoadSystemItem { id: Id },
    Access { field: u32 },
    GetType,
    Call { which: usize },
    SystemCall { id: Id },
    Return,
    Swap { with: usize },
    Pull { which: usize },
    Pop { count: usize, offset: usize },
    Copy { which: usize },
    Jump { to: usize, check: Option<bool> },
}

#[derive(Debug)]
pub struct CrateDeclaration {
    pub id: CrateId,
    pub dependencies: Vec<(CrateId, NonZero<u32>)>,
    pub items: Vec<ConstItem>,
    pub functions: Vec<FunctionDeclaration>,
    pub implementations: HashMap<ImplId, Implementation>,
}

#[derive(Debug, Clone)]
pub enum Integer {
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
}

#[derive(Debug, Clone)]
pub enum ConstItem {
    Integer(Integer),
    Float(f64),
    Char(char),
    String(String),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub code: Vec<Op>,
    pub mod_scope: usize,
}

#[derive(Debug, Clone)]
pub struct FunctionDeclaration {
    pub code: Vec<Op>,
}

#[derive(Debug, Clone)]
pub enum SystemObj {
    Bool(bool),
    Void,
    Type(ObjType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveIntType {
    I16,
    I32,
    I64,
    I128,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveObjType {
    Float,
    Integer(PrimitiveIntType),
    String,
    Char,
    Bool,
    Void,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjType {
    Declared(Id),
    Primitive(PrimitiveObjType),
    Function,
    Type,
}

#[derive(Debug, Clone)]
pub enum Object {
    Fundamental(ConstItem),
    Function(Function),
    Constructed { fields: Vec<OPtr>, id: Id },
    System(SystemObj)
}

impl Object {
    pub fn get_type(&self) -> ObjType {
        match self {
            Object::System(SystemObj::Void) => ObjType::Primitive(PrimitiveObjType::Void),
            Object::System(SystemObj::Bool(..)) => ObjType::Primitive(PrimitiveObjType::Bool),
            Object::System(SystemObj::Type(..)) => ObjType::Type,
            Object::Function(..) => ObjType::Function,
            Object::Constructed { id, .. } => ObjType::Declared(*id),
            Object::Fundamental(ConstItem::String(..)) => ObjType::Primitive(PrimitiveObjType::String),
            Object::Fundamental(ConstItem::Char(..)) => ObjType::Primitive(PrimitiveObjType::Char),
            Object::Fundamental(ConstItem::Float(..)) => ObjType::Primitive(PrimitiveObjType::Float),
            Object::Fundamental(ConstItem::Integer(Integer::I16(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::I16)),
            Object::Fundamental(ConstItem::Integer(Integer::I32(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::I32)),
            Object::Fundamental(ConstItem::Integer(Integer::I64(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::I64)),
            Object::Fundamental(ConstItem::Integer(Integer::I128(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::I128)),
            Object::Fundamental(ConstItem::Integer(Integer::U16(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::U16)),
            Object::Fundamental(ConstItem::Integer(Integer::U32(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::U32)),
            Object::Fundamental(ConstItem::Integer(Integer::U64(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::U64)),
            Object::Fundamental(ConstItem::Integer(Integer::U128(..))) => ObjType::Primitive(PrimitiveObjType::Integer(PrimitiveIntType::U128)),
        }
    }
}

#[derive(Debug)]
pub struct Implementation {
    pub func: HashMap<u32, Function>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplId {
    pub of: u32,
    pub r#for: ObjType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateId {
    pub name: u32,  // hash of name
    pub ver: u32,   // hash of version
}


impl CrateId {
    pub fn new(name: &str, ver: (u16, u16, u16)) -> Self {
        Self {
            name: Self::hash_name(name),
            ver: Self::hash_ver(ver),
        }
    }

    // based on djb33
    // fixme test for collisions
    // xxx measure perfomance, just out of curiosity
    pub fn hash_name(name: &str) -> u32 {
        const INIT: u32 = 2147483647;
        const COEF: u32 = 134217689;
        const FINAL: u32 = 33554393;

        name.as_bytes().iter().copied()
            .fold(INIT, |h, b| h.wrapping_mul(b as u32).wrapping_mul(33) ^ (b as u32).wrapping_mul(COEF))
            .wrapping_mul(FINAL)
    }

    // i think i have spend too much time on this, based on djb33
    // fixme actually test it against collisions
    // xxx measure perfomance, just out of curiosity
    pub fn hash_ver(ver: (u16, u16, u16)) -> u32 {
        let ver = unsafe { std::mem::transmute::<_, [u8; 6]>(ver) };

        const INIT: u32 = 2147483647;
        const COEFS_H: [u32; 6] = [16777213, 8388593, 67108729, 33564343, 4194301, 2097143];
        const COEFS_B: [u32; 6] = [1073741789, 536870909, 268447147, 134217689, 67108859, 33554393];

        debug_assert_eq!(ver.len(), COEFS_H.len());
        debug_assert_eq!(ver.len(), COEFS_B.len());

        ver.into_iter().zip(COEFS_H.into_iter()).zip(COEFS_B.into_iter())
            .fold(INIT, |h, ((b, k_h), k_b)| h.wrapping_mul(k_h) ^ (b as u32).wrapping_mul(k_b))
    }
}


impl From<(Box<str>, (u16, u16, u16))> for CrateId {
    fn from((name, ver): (Box<str>, (u16, u16, u16))) -> Self {
        Self::new(&name, ver)
    }
}


#[derive(Debug)]
pub struct Module {
    pub id: CrateId,
    pub mod_table: HashMap<NonZero<u32>, usize>,
    pub items: Vec<ConstItem>,
    pub functions: Vec<Function>,
    pub implementations: HashMap<ImplId, Implementation>
}

#[derive(Debug, Clone, Copy)]
pub struct OPtr {
    pub id: NonZero<usize>,
    pub index: usize,
}

#[derive(Debug)]
pub struct ObjectCell {
    object: Object,
    id: NonZero<usize>,
    refcount: usize,
}

// todo resolution of circular things
#[derive(Debug, Default)]
struct Memory {
    last_id: usize,
    cells: Vec<Option<ObjectCell>>,
}

impl Memory {
    fn find_first_empty(&self) -> Option<usize> {
        self.cells.iter().position(Option::is_none)
    }

    fn alloc_obj_cell(&mut self, object: Object) -> (Option<ObjectCell>, NonZero<usize>) {
        // tbh we shouldnt ever run out of them
        // also skip 0 when overflowing
        self.last_id = self.last_id.checked_add(1).unwrap_or(1);
        let id = NonZero::new(self.last_id).expect("0 + 1 = 1");
        (
            Some(ObjectCell {
                refcount: 0,
                id,
                object,
            }),
            id
        )
    }

    pub fn alloc(&mut self, obj: Object) -> OPtr {
        let (cell, id) = self.alloc_obj_cell(obj);

        if let Some(i) = self.find_first_empty() {
            self.cells[i] = cell;
            OPtr { index: i, id }
        } else {
            self.cells.push(cell);
            OPtr { index: self.cells.len() - 1, id }
        }
    }

    fn get_cell(&self, optr: OPtr) -> &ObjectCell {
        let cell = self.cells[optr.index].as_ref().expect("it should be not none");
        if cell.id != optr.id {
            panic!("invalid id; hanging ref");
        };
        cell
    }

    fn get_cell_mut(&mut self, optr: OPtr) -> &mut ObjectCell {
        let cell = self.cells[optr.index].as_mut().expect("it should be not none");
        if cell.id != optr.id {
            panic!("invalid id; hanging ref");
        };
        cell
    }

    pub fn reg_ref(&mut self, optr: OPtr) {
        let cell = self.get_cell_mut(optr);
        cell.refcount += 1;
    }

    pub fn dereg_ref(&mut self, optr: OPtr) {
        let cell = self.get_cell_mut(optr);
        if let Some(dec) = cell.refcount.checked_sub(1) {
            cell.refcount = dec;
        } else {
            match &cell.object {
                Object::Constructed { fields, .. } => {
                    for field in fields.clone() {  // fixme resolve this clone
                        self.dereg_ref(field);
                    };
                },
                _ => {},
            };
            self.cells[optr.index] = None;
        }
    }
}

impl Index<OPtr> for Memory {
    type Output = Object;

    fn index(&self, index: OPtr) -> &Self::Output {
        &self.get_cell(index).object
    }
}

#[derive(Debug)]
struct FunctionScope {
    func: Function,  // todo as ref
    cur_instr: usize,
}

type FStep = ControlFlow<(), Option<FunctionScope>>;

impl FunctionScope {
    pub fn new(func: Function) -> Self {
        Self { func, cur_instr: 0 }
    }
    
    fn get_mod_i(&self, id: Id, vm: &Vm) -> usize {
        // fixme remove an indirection through vm
        id.space.map_or(self.func.mod_scope, |s| vm.modules[self.func.mod_scope].mod_table[&s])
    }
    
    fn get_mod<'v>(&self, id: Id, vm: &'v Vm) -> &'v Module {
        &vm.modules[self.get_mod_i(id, vm)]
    }

    pub fn step(&mut self, vm: &mut Vm) -> FStep {
        let mut res = ControlFlow::Continue(None);
        let mut inc = true;
        match &self.func.code[self.cur_instr] {
            Op::Pack { id, count } => {
                vm.push_new(Object::Constructed { id: *id, fields: vm.stack[..*count].to_vec() });
            },
            Op::LoadConstItem { id } => {
                vm.push_new(Object::Fundamental(self.get_mod(*id, vm).items[id.item as usize].clone()));
            },
            Op::LoadFunction { id } => {
                vm.push_new(Object::Function(self.get_mod(*id, vm).functions[id.item as usize].clone()));
            },
            Op::LoadImplementation { id, func } => {
                let top_obj = &vm.memory[vm.stack[0]];

                vm.push_new(Object::Function(self.get_mod(*id, vm).implementations[&ImplId { of: id.item, r#for: top_obj.get_type() }].func[func].clone()));
            },
            Op::LoadSystemItem { id } => {
                assert_eq!(id.space, None);
                vm.push_new(match id.item {
                    0 => Object::System(SystemObj::Bool(false)),
                    1 => Object::System(SystemObj::Bool(true)),
                    2 => Object::System(SystemObj::Void),
                    _ => panic!("invalid id for system item"),
                });
            },
            Op::Access { field } => {
                let top_obj = &vm.memory[vm.stack[0]];
                match top_obj {
                    Object::Constructed { fields, .. } => {
                        let field = fields[*field as usize];
                        vm.memory.reg_ref(field);
                        vm.stack.push(field);
                    },
                    _ => panic!("doesnt have any fields")
                }
            },
            Op::GetType => {
                let top_obj = &vm.memory[vm.stack[0]];
                vm.push_new(Object::System(SystemObj::Type(top_obj.get_type())));
            },
            Op::Call { which } => {
                match &vm.memory[vm.stack[*which]] {
                    Object::Function(func) => res = ControlFlow::Continue(Some(FunctionScope::new(func.clone()))),
                    _ => panic!("cant call non functions")
                }
            },
            Op::SystemCall { id } => {
                assert_eq!(id.space, None);
                match id.item {
                    0 => panic!("panic."),
                    1 => println!("{}", match &vm.memory[vm.stack[0]] {
                        Object::Fundamental(ConstItem::String(s)) => s,
                        _ => panic!()
                    }),
                    _ => panic!("unknown syscall")
                }
            },
            Op::Return => {
                res = ControlFlow::Break(());
            },
            Op::Swap { with } => {
                vm.stack.swap(*with);
            },
            Op::Pull { which } => {
                vm.stack.pull(*which)
            },
            Op::Pop { offset, count } => {
                for popped in vm.stack.pop_many(*offset, *count) {
                    vm.memory.dereg_ref(popped);
                };
            },
            Op::Copy { which } => {
                let optr = vm.stack[*which];
                vm.memory.reg_ref(optr);
                vm.stack.push(optr);
            },
            Op::Jump { to, check} => {
                if let Some(expected) = check {
                    match &vm.memory[vm.stack[0]] {
                        Object::System(SystemObj::Bool(b)) if b == expected => {
                            self.cur_instr = *to;
                            inc = false;
                        },
                        Object::System(SystemObj::Bool(..)) => {},
                        _ => panic!("can check only with bools")
                    }
                } else {
                    self.cur_instr = *to;
                    inc = false;
                };
            },
        };
        
        if inc {
            self.cur_instr += 1;
        };

        res
    }
}

#[derive(Debug, Default)]
pub struct Stack {
    inner: Vec<OPtr>,
}

impl Stack {
    pub fn push(&mut self, optr: OPtr) {
        self.inner.push(optr)
    }

    pub fn pop(&mut self) -> OPtr {
        self.inner.pop().expect("if pop is called, it should be there")
    }

    pub fn swap(&mut self, with: usize) {
        let len = self.inner.len();
        self.inner.swap(len-1, len-1-with)
    }
    
    pub fn pull(&mut self, which: usize) {
        let item = self.inner.remove(self.inner.len()-1-which);
        self.inner.push(item);
    }

    pub fn pop_many(&mut self, offset: usize, count: usize) -> impl Iterator<Item = OPtr> + use<'_> {
        self.inner.drain(self.inner.len()-count-offset .. self.inner.len()-offset)
    }
}

impl Index<usize> for Stack {
    type Output = OPtr;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[self.inner.len()-index-1]
    }
}

impl Index<Range<usize>> for Stack {
    type Output = [OPtr];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        let len = self.inner.len();
        &self.inner[len-index.end .. len-index.start]
    }
}

impl Index<RangeTo<usize>> for Stack {
    type Output = [OPtr];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.inner[self.inner.len()-index.end..]
    }
}

#[derive(Default)]
pub struct Vm {
    mod_map: HashMap<CrateId, usize>,
    modules: Vec<Module>,
    memory: Memory,
    stack: Stack,
    scopes: Vec<FunctionScope>,
}

#[derive(Debug)]
pub enum StepError {
    
}

impl Vm {
    fn push_new(&mut self, obj: Object) {
        if let Object::Constructed { fields, .. } = &obj {
            for field in fields {
                self.memory.reg_ref(*field);
            };
        };

        let optr = self.memory.alloc(obj);

        self.memory.reg_ref(optr);
        self.stack.push(optr);
    }
    
    pub fn load_crate(&mut self, crate_decl: CrateDeclaration) -> usize {
        let index = self.modules.len();

        let mod_table = crate_decl.dependencies.into_iter()
            .map(|(id, space)| (space, self.mod_map[&id]))
            .collect::<HashMap<_, _>>();
        
        let module = Module {
            id: crate_decl.id,
            functions: crate_decl.functions.into_iter().map(|decl| Function { code: decl.code, mod_scope: index }).collect(),
            items: crate_decl.items,
            implementations: crate_decl.implementations,
            mod_table, 
        };

        self.mod_map.insert(module.id, index);
        self.modules.push(module);
        index
    }
    
    pub fn call(&mut self, mod_id: usize, func_id: u32) {
        self.scopes.push(FunctionScope::new(self.modules[mod_id].functions[func_id as usize].clone()))
    }
    
    pub fn step(&mut self) -> Option<Result<(), StepError>> {
        let mut cur_scope = self.scopes.pop()?;
        match cur_scope.step(self) {
            ControlFlow::Continue(new_scope) => {
                self.scopes.push(cur_scope);
                if let Some(scope) = new_scope {
                    self.scopes.push(scope);
                };
            },
            ControlFlow::Break(()) => {},
        };
        Some(Ok(()))
    }

    pub fn run_till_end(&mut self) -> (Result<(), StepError>, u64) {
        let mut c = 0u64;
        (loop {
            match self.step() {
                None => break Ok(()),
                Some(Ok(())) => { c = c.saturating_add(1) },
                Some(Err(err)) => break Err(err)
            };
        }, c)
    }
}
