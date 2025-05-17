#![feature(let_chains)]

use crate::compiler::{Compiler, CrateSource, Path};

pub mod parser;
mod vm;
#[macro_use]
mod compiler;

fn main() {
    let parsed = parser::ModuleParser::new().parse(
r#"
fnc println |s: %str|
asm {
    syscall println;
    return;
};

fnc always_true || -> %bool {
    true
};

fnc always_false || -> %bool {
    false
};

fnc entry || {
    println("hello");

    #if always_true() {
        println("this checks out!");
    };

    #if always_false() {
        println("uh?");
    } else {
        println("this is too :sunglasses:");
    };
};
"#
    ).unwrap();
    println!("{parsed:#?}");
    
    let mut compiler = Compiler::default();

    compiler.load_crate(
        CrateSource {
            id: ("example".into(), (0, 0, 0)),
            deps: vec![],
            mods: vec![(path!(), parsed)],
        }
    );
    
    // println!("{compiler:#?}");  // todo make actually readable IR display impl
    
    let (crates, entry_func) = compiler.compile(Some(path!(example @ entry))).unwrap();

    println!("{crates:#?}");
    
    let mut vm = vm::Vm::default();

    let crate_id = vm.load_crate(crates.into_iter().next().unwrap());

    vm.call(crate_id, entry_func.unwrap().1);

    vm.run_till_end().0.unwrap();
}
