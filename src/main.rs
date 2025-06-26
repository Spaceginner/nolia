use compiler::{lcr::Path, Compiler, CrateSource, path};
use syntax::{lexer::Lexer, parser::ModuleParser};

fn main() {
    let file = std::fs::read_to_string("test.nol").unwrap();
    let lexer = Lexer::new(&file);
    let parsed = ModuleParser::new().parse(lexer).unwrap();

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
