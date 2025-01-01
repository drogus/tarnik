use crate::{
    InstructionsList, InstructionsListWrapped, TypeDefinition, WasmType, WatFunction,
    WatInstruction, WatModule,
};
use std::{cell::RefCell, rc::Rc};

// A structure able to traverse WAT AST tree and modify it. I don't really like the current
// implementation, but it was the simplest thing to do for now. The problem with the way it's done
// at the moment is that
#[derive(Debug, Clone)]
pub struct InstructionsCursor<'a> {
    module: &'a WatModule,
    instructions: InstructionsListWrapped,
    position: usize,
}

impl<'a> InstructionsCursor<'a> {
    pub fn new(module: &'a WatModule, instructions: InstructionsListWrapped) -> Self {
        Self {
            module,
            instructions,
            position: 0,
        }
    }

    pub fn replace_current(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<WatInstruction> {
        if self.position < self.instructions.borrow().len() {
            let old = self.instructions.borrow_mut()[self.position].clone();

            // Replace the current instruction with the first new one
            self.instructions.borrow_mut()[self.position] = new_instructions[0].clone();

            // Insert any remaining instructions after the current position
            if new_instructions.len() > 1 {
                let remaining = &new_instructions[1..];
                self.instructions.borrow_mut().splice(
                    self.position + 1..self.position + 1,
                    remaining.iter().cloned(),
                );
            }

            Some(old)
        } else {
            None
        }
    }

    pub fn insert_after_current(&mut self, new_instructions: Vec<WatInstruction>) {
        if self.position < self.instructions.borrow().len() {
            // Insert all new instructions after the current position
            self.instructions.borrow_mut().splice(
                self.position + 1..self.position + 1,
                new_instructions.iter().cloned(),
            );
        }
    }

    pub fn from_function(module: &'a WatModule, function: &WatFunction) -> Self {
        Self::new(module, function.body.clone())
    }

    pub fn current_instruction(&self) -> Option<WatInstruction> {
        if self.position < self.instructions.borrow().len() {
            Some(self.instructions.borrow()[self.position].clone())
        } else {
            None
        }
    }

    pub fn next(&mut self) -> bool {
        if self.position + 1 < self.instructions.borrow().len() {
            self.position += 1;
            true
        } else {
            false
        }
    }

    pub fn previous(&mut self) -> bool {
        if self.position > 0 {
            self.position -= 1;
            true
        } else {
            false
        }
    }

    fn instruction_stack_effect(&self, instruction: &WatInstruction) -> (usize, isize) {
        match instruction {
            WatInstruction::I32Const(_)
            | WatInstruction::I64Const(_)
            | WatInstruction::F32Const(_)
            | WatInstruction::F64Const(_) => (0, 1),
            WatInstruction::I32Add
            | WatInstruction::I64Add
            | WatInstruction::F32Add
            | WatInstruction::F64Add
            | WatInstruction::I32Sub
            | WatInstruction::I64Sub
            | WatInstruction::F32Sub
            | WatInstruction::F64Sub
            | WatInstruction::I32Mul
            | WatInstruction::I64Mul
            | WatInstruction::F32Mul
            | WatInstruction::F64Mul
            | WatInstruction::I32DivS
            | WatInstruction::I64DivS
            | WatInstruction::F32Div
            | WatInstruction::F64Div
            | WatInstruction::I32And
            | WatInstruction::I64And
            | WatInstruction::I32Or
            | WatInstruction::I64Or
            | WatInstruction::I32Xor
            | WatInstruction::I64Xor => (2, 1),
            WatInstruction::Call(name) => {
                if let Some(func) = self
                    .module
                    .functions
                    .iter()
                    .find(|f| &format!("${}", f.name) == name)
                {
                    (
                        func.params.len(),
                        if func.results.is_empty() { 0 } else { 1 },
                    )
                } else {
                    (0, 0)
                }
            }
            WatInstruction::LocalGet(_) => (0, 1),
            WatInstruction::LocalSet(_) => (1, 0),
            WatInstruction::LocalTee(_) => (1, 1),
            WatInstruction::GlobalGet(_) => (0, 1),
            WatInstruction::GlobalSet(_) => (1, 0),
            WatInstruction::RefNull(_) => (0, 1),
            WatInstruction::Nop => (0, 0),
            WatInstruction::Local { .. } => (0, 0),
            WatInstruction::CallRef(name) => {
                if let Some(func) = self
                    .module
                    .functions
                    .iter()
                    .find(|f| &format!("${}", f.name) == name)
                {
                    (
                        func.params.len(),
                        if func.results.is_empty() { 0 } else { 1 },
                    )
                } else {
                    (0, 0)
                }
            }
            WatInstruction::F32Neg => (1, 1),
            WatInstruction::F64Neg => (1, 1),
            WatInstruction::I32Eqz => (1, 1),
            WatInstruction::I64Eqz => (1, 1),
            WatInstruction::F32Eqz => (1, 1),
            WatInstruction::F64Eqz => (1, 1),
            WatInstruction::I32Eq => (2, 1),
            WatInstruction::I64Eq => (2, 1),
            WatInstruction::F32Eq => (2, 1),
            WatInstruction::F64Eq => (2, 1),
            WatInstruction::I32Ne => (2, 1),
            WatInstruction::I64Ne => (2, 1),
            WatInstruction::F32Ne => (2, 1),
            WatInstruction::F64Ne => (2, 1),
            WatInstruction::I32DivU => (2, 1),
            WatInstruction::I64DivU => (2, 1),
            WatInstruction::I32RemS => (2, 1),
            WatInstruction::I64RemS => (2, 1),
            WatInstruction::I64RemU => (2, 1),
            WatInstruction::I32RemU => (2, 1),
            WatInstruction::I32LtS => (2, 1),
            WatInstruction::I64LtS => (2, 1),
            WatInstruction::I32LtU => (2, 1),
            WatInstruction::I64LtU => (2, 1),
            WatInstruction::F32Lt => (2, 1),
            WatInstruction::F64Lt => (2, 1),
            WatInstruction::I32LeS => (2, 1),
            WatInstruction::I64LeS => (2, 1),
            WatInstruction::I32LeU => (2, 1),
            WatInstruction::I64LeU => (2, 1),
            WatInstruction::F32Le => (2, 1),
            WatInstruction::F64Le => (2, 1),
            WatInstruction::I32GeS => (2, 1),
            WatInstruction::I64GeS => (2, 1),
            WatInstruction::I32GeU => (2, 1),
            WatInstruction::I64GeU => (2, 1),
            WatInstruction::F32Ge => (2, 1),
            WatInstruction::F64Ge => (2, 1),
            WatInstruction::I32GtS => (2, 1),
            WatInstruction::I64GtS => (2, 1),
            WatInstruction::I32GtU => (2, 1),
            WatInstruction::I64GtU => (2, 1),
            WatInstruction::F32Gt => (2, 1),
            WatInstruction::F64Gt => (2, 1),
            WatInstruction::I32Shl => (1, 1),
            WatInstruction::I64Shl => (1, 1),
            WatInstruction::I32ShrS => (1, 1),
            WatInstruction::I64ShrS => (1, 1),
            WatInstruction::I32ShrU => (1, 1),
            WatInstruction::I64ShrU => (1, 1),
            WatInstruction::I64ExtendI32S => (1, 1),
            WatInstruction::I32WrapI64 => (1, 1),
            WatInstruction::F64PromoteF32 => (1, 1),
            WatInstruction::F32DemoteF64 => (1, 1),
            WatInstruction::F32ConvertI32S => (1, 1),
            WatInstruction::F32ConvertI32U => (1, 1),
            WatInstruction::F32ConvertI64S => (1, 1),
            WatInstruction::F32ConvertI64U => (1, 1),
            WatInstruction::F64ConvertI32S => (1, 1),
            WatInstruction::F64ConvertI32U => (1, 1),
            WatInstruction::F64ConvertI64S => (1, 1),
            WatInstruction::F64ConvertI64U => (1, 1),
            WatInstruction::I32TruncF32S => (1, 1),
            WatInstruction::I32TruncF32U => (1, 1),
            WatInstruction::I32TruncF64S => (1, 1),
            WatInstruction::I32TruncF64U => (1, 1),
            WatInstruction::I64TruncF32S => (1, 1),
            WatInstruction::I64TruncF32U => (1, 1),
            WatInstruction::I64TruncF64S => (1, 1),
            WatInstruction::I64TruncF64U => (1, 1),
            WatInstruction::I32ReinterpretF32 => (1, 1),
            WatInstruction::F32ReinterpretI32 => (1, 1),
            WatInstruction::I64ReinterpretF64 => (1, 1),
            WatInstruction::F64ReinterpretI64 => (1, 1),
            WatInstruction::I31GetS => (1, 1),
            WatInstruction::I31GetU => (1, 1),
            WatInstruction::I32Store(_) => (2, 0),
            WatInstruction::I64Store(_) => (2, 0),
            WatInstruction::F32Store(_) => (2, 0),
            WatInstruction::F64Store(_) => (2, 0),
            WatInstruction::I32Store8(_) => (2, 0),
            WatInstruction::I32Store16(_) => (2, 0),
            WatInstruction::I64Store8(_) => (2, 0),
            WatInstruction::I64Store16(_) => (2, 0),
            WatInstruction::I64Store32(_) => (2, 0),
            WatInstruction::I32Load(_) => (1, 1),
            WatInstruction::I64Load(_) => (1, 1),
            WatInstruction::F32Load(_) => (1, 1),
            WatInstruction::F64Load(_) => (1, 1),
            WatInstruction::I32Load8S(_) => (1, 1),
            WatInstruction::I32Load8U(_) => (1, 1),
            WatInstruction::I32Load16S(_) => (1, 1),
            WatInstruction::I32Load16U(_) => (1, 1),
            WatInstruction::I64Load8S(_) => (1, 1),
            WatInstruction::I64Load8U(_) => (1, 1),
            WatInstruction::I64Load16S(_) => (1, 1),
            WatInstruction::I64Load16U(_) => (1, 1),
            WatInstruction::I64Load32S(_) => (1, 1),
            WatInstruction::I64Load32U(_) => (1, 1),
            WatInstruction::StructNew(name) => {
                if let Some(t) = self.module.types.iter().find_map(|t| match t {
                    TypeDefinition::Type(n, type_) if n == name => Some(type_),
                    _ => None,
                }) {
                    if let WasmType::Struct(fields) = t {
                        (fields.len(), 1)
                    } else {
                        unreachable!()
                    }
                } else {
                    (0, 1)
                }
            }
            WatInstruction::StructGet(_, _) => (1, 1),
            WatInstruction::StructSet(_, _) => (2, 0),
            WatInstruction::ArrayNew(_) => (2, 1),
            WatInstruction::ArrayNewFixed(_, count) => (*count as usize, 1),
            WatInstruction::ArrayLen => (1, 1),
            WatInstruction::ArrayGet(_) => (2, 1),
            WatInstruction::ArrayGetU(_) => (2, 1),
            WatInstruction::ArraySet(_) => (3, 0),
            WatInstruction::RefCast(_) => (1, 1),
            WatInstruction::RefTest(_) => (1, 1),
            // TODO: Do we even use this?
            WatInstruction::Ref(_) => (0, 1),
            WatInstruction::RefFunc(_) => (0, 1),
            // TODO: Do we even use this? It's not a realy instruction
            WatInstruction::Type(_) => (0, 0),
            // TODO: this will break if we allow more than one return types
            WatInstruction::Return => (1, 0),
            WatInstruction::ReturnCall(name) => {
                if let Some(func) = self
                    .module
                    .functions
                    .iter()
                    .find(|f| &format!("${}", f.name) == name)
                {
                    (
                        func.params.len(),
                        if func.results.is_empty() { 0 } else { 1 },
                    )
                } else {
                    (0, 0)
                }
            }
            WatInstruction::Block { .. } => (0, 0),
            WatInstruction::Loop { .. } => (0, 0),
            WatInstruction::If { .. } => (1, 0),
            WatInstruction::BrIf(_) => (1, 0),
            WatInstruction::Br(_) => (0, 0),
            WatInstruction::Empty => (0, 0),
            // TODO: this will break if we allow more than one return types
            WatInstruction::Log => (1, 0),
            // TODO: this will break if we allow more than one return types
            WatInstruction::Identifier(_) => (0, 0),
            WatInstruction::Drop => (1, -1),
            WatInstruction::RefI31 => (1, 1),
            // TODO: we need to check how many arguments does the exception type need
            WatInstruction::Throw(_) => (1, 0),
            WatInstruction::Try { .. } => (0, 0),
            WatInstruction::Catch(_, _) => (0, 0),
            WatInstruction::CatchAll(_) => (0, 0),
            WatInstruction::RefEq => (2, 1),
        }
    }

    // TODO: can I somehow return Vec<Vec<&WatInstruction>> here?
    pub fn get_arguments_instructions(&'a self, count: usize) -> Vec<Vec<WatInstruction>> {
        let mut result = Vec::new();
        let mut stack_size = 0isize;
        // TODO: get rid of these unwraps
        let mut pos = self.position;
        let instructions = self.instructions.borrow();

        let mut current_argument = Vec::new();
        let mut current_argument_stack_size = 0isize;
        while stack_size < count as isize && pos > 0 {
            pos -= 1;
            let instruction = &instructions[pos];
            let (takes, produces) = self.instruction_stack_effect(instruction);

            current_argument.push(instruction.clone());
            stack_size += produces;
            stack_size -= takes as isize;

            current_argument_stack_size += produces;
            current_argument_stack_size -= takes as isize;

            let can_finish =
                pos == 0 || self.instruction_stack_effect(&instructions[pos - 1]).1 != 0;

            if current_argument_stack_size == 1 && can_finish {
                current_argument.reverse();
                result.push(current_argument);
                current_argument_stack_size = 0;
                current_argument = Vec::new();
            }
        }

        if !current_argument.is_empty() {
            result.push(current_argument);
        }

        result.reverse();
        result
    }

    pub fn get_block_instructions(&self) -> InstructionsListWrapped {
        let current = self.current_instruction().unwrap();

        match current {
            WatInstruction::Block { instructions, .. } => instructions,
            WatInstruction::Loop { instructions, .. } => instructions,
            WatInstruction::If { then, .. } => then,
            _ => unreachable!(),
        }
    }

    pub fn enter_block(&'a self) -> Vec<InstructionsCursor<'a>> {
        let mut cursors = Vec::new();

        if let Some(current) = self.current_instruction() {
            match current {
                WatInstruction::Block { instructions, .. } => {
                    cursors.push(InstructionsCursor::new(self.module, instructions));
                }
                WatInstruction::Loop { instructions, .. } => {
                    cursors.push(InstructionsCursor::new(self.module, instructions));
                }
                WatInstruction::If { then, r#else } => {
                    cursors.push(InstructionsCursor::new(self.module, then));
                    if let Some(else_block) = r#else {
                        cursors.push(InstructionsCursor::new(self.module, else_block));
                    }
                }
                WatInstruction::Try {
                    try_block,
                    catches,
                    catch_all,
                } => {
                    cursors.push(InstructionsCursor::new(self.module, try_block));

                    for (_, catch_block) in catches {
                        cursors.push(InstructionsCursor::new(self.module, catch_block));
                    }

                    if let Some(catch_all_block) = catch_all {
                        cursors.push(InstructionsCursor::new(
                            self.module,
                            Rc::new(RefCell::new(catch_all_block.clone())),
                        ));
                    }
                }
                WatInstruction::Catch(_, instructions) => {
                    cursors.push(InstructionsCursor::new(self.module, instructions));
                }
                WatInstruction::CatchAll(instructions) => {
                    cursors.push(InstructionsCursor::new(self.module, instructions));
                }
                _ => {}
            }
        }

        cursors
    }

    pub fn get_call_arguments(&self) -> Option<Vec<Vec<WatInstruction>>> {
        match self.current_instruction() {
            Some(WatInstruction::Call(name)) => {
                if let Some(func) = self
                    .module
                    .functions
                    .iter()
                    .find(|f| format!("${}", f.name) == name)
                {
                    Some(self.get_arguments_instructions(func.params.len()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn replace_current_call_with_arguments(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        match self.current_instruction() {
            Some(WatInstruction::Call(name)) => {
                if let Some(func) = self
                    .module
                    .functions
                    .iter()
                    .find(|f| format!("${}", f.name) == name)
                {
                    let param_count = func.params.len();
                    let mut old_instructions = Vec::new();
                    let mut stack_size = 0isize;
                    let mut pos = self.position;
                    let original_position = pos;

                    let mut instructions = self.instructions.borrow_mut();
                    // Collect the call instruction itself
                    old_instructions.push(instructions[pos].clone());

                    // Collect all argument instructions
                    while stack_size < param_count as isize && pos > 0 {
                        pos -= 1;
                        let instruction = instructions[pos].clone();
                        let (takes, produces) = self.instruction_stack_effect(&instruction);

                        old_instructions.push(instruction);
                        stack_size += produces;
                        stack_size -= takes as isize;
                    }

                    let new_instructions_len = new_instructions.len();

                    // Remove all old instructions
                    instructions.splice(pos..=original_position, new_instructions);

                    // Update position to point to the last new instruction
                    self.position = pos + new_instructions_len - 1;

                    old_instructions.reverse();
                    Some(old_instructions)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WasmType;

    fn create_test_module() -> WatModule {
        let mut module = WatModule::new();

        // Add a test function that takes 2 i32 parameters
        let mut func = WatFunction::new("foo");
        func.add_param("a", &WasmType::I32);
        func.add_param("b", &WasmType::I32);
        module.add_function(func);

        module
    }

    #[test]
    fn test_cursor_navigation() {
        let module = create_test_module();
        let instructions = vec![
            WatInstruction::I32Const(1),
            WatInstruction::I32Const(2),
            WatInstruction::I32Add,
        ];

        let mut cursor = InstructionsCursor::new(&module, Rc::new(RefCell::new(instructions)));

        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
        assert!(cursor.next());
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
        assert!(cursor.next());
        assert_eq!(cursor.current_instruction(), Some(WatInstruction::I32Add));
        assert!(!cursor.next());

        assert!(cursor.previous());
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
    }

    #[test]
    fn test_block_navigation() {
        let module = create_test_module();
        let block_instructions = Rc::new(RefCell::new(vec![
            WatInstruction::I32Const(1),
            WatInstruction::I32Const(2),
        ]));
        let instructions = Rc::new(RefCell::new(vec![WatInstruction::Block {
            label: "test".to_string(),
            signature: Default::default(),
            instructions: block_instructions,
        }]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let block_cursor = &mut cursor.enter_block()[0];

        assert_eq!(
            block_cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
        assert!(block_cursor.next());
        assert_eq!(
            block_cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
    }

    #[test]
    fn test_function_arguments() {
        let module = create_test_module();
        let instructions = vec![
            WatInstruction::I32Const(10),
            WatInstruction::I32Const(1),
            WatInstruction::I32Const(2),
            WatInstruction::I32Add,
            WatInstruction::Call("$foo".to_string()),
        ];

        let mut cursor = InstructionsCursor::new(&module, Rc::new(RefCell::new(instructions)));
        while cursor.current_instruction() != Some(WatInstruction::Call("$foo".to_string())) {
            cursor.next();
        }

        let parts = cursor.get_call_arguments().unwrap();
        assert_eq!(parts.len(), 2);

        assert_eq!(parts[0].len(), 1);
        assert_eq!(parts[0][0], WatInstruction::I32Const(10));

        assert_eq!(parts[1].len(), 3);
        assert_eq!(parts[1][0], WatInstruction::I32Const(1));
        assert_eq!(parts[1][1], WatInstruction::I32Const(2));
        assert_eq!(parts[1][2], WatInstruction::I32Add);
    }

    #[test]
    fn test_function_arguments_with_set_get() {
        let module = create_test_module();
        let instructions = vec![
            WatInstruction::I32Const(10),
            WatInstruction::I32Const(1),
            WatInstruction::LocalSet("$var".to_string()),
            WatInstruction::LocalGet("$var".to_string()),
            WatInstruction::I32Const(2),
            WatInstruction::I32Add,
            WatInstruction::Call("$foo".to_string()),
        ];

        let mut cursor = InstructionsCursor::new(&module, Rc::new(RefCell::new(instructions)));
        while cursor.current_instruction() != Some(WatInstruction::Call("$foo".to_string())) {
            cursor.next();
        }

        let parts = cursor.get_call_arguments().unwrap();
        assert_eq!(parts.len(), 2);

        assert_eq!(parts[0].len(), 1);
        assert_eq!(parts[0][0], WatInstruction::I32Const(10));

        assert_eq!(parts[1].len(), 5);
        assert_eq!(parts[1][0], WatInstruction::I32Const(1));
        assert_eq!(parts[1][1], WatInstruction::LocalSet("$var".to_string()));
        assert_eq!(parts[1][2], WatInstruction::LocalGet("$var".to_string()));
        assert_eq!(parts[1][3], WatInstruction::I32Const(2));
        assert_eq!(parts[1][4], WatInstruction::I32Add);
    }

    #[test]
    fn test_nested_blocks() {
        let module = create_test_module();
        let inner_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(1)]));
        let outer_block = Rc::new(RefCell::new(vec![
            WatInstruction::Block {
                label: "inner".to_string(),
                signature: Default::default(),
                instructions: inner_block,
            },
            WatInstruction::I32Const(2),
        ]));
        let instructions = Rc::new(RefCell::new(vec![WatInstruction::Block {
            label: "outer".to_string(),
            signature: Default::default(),
            instructions: outer_block,
        }]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let outer = &cursor.enter_block()[0];
        let inner = &outer.enter_block()[0];

        assert_eq!(
            inner.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
    }

    #[test]
    fn test_if_block() {
        let module = create_test_module();
        let then_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(1)]));
        let else_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(2)]));
        let instructions = Rc::new(RefCell::new(vec![WatInstruction::If {
            then: then_block,
            r#else: Some(else_block),
        }]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let if_cursor = &cursor.enter_block()[0];
        let else_cursor = &cursor.enter_block()[1];

        assert_eq!(
            if_cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );

        assert_eq!(
            else_cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
    }

    #[test]
    fn test_replace_current_call_with_arguments() {
        let module = create_test_module();
        let instructions = Rc::new(RefCell::new(vec![
            WatInstruction::I32Const(999),
            WatInstruction::I32Const(1),
            WatInstruction::I32Const(2),
            WatInstruction::I32Add,
            WatInstruction::I32Const(10),
            WatInstruction::Call("$foo".to_string()),
            WatInstruction::I32Const(888),
        ]));

        let mut cursor = InstructionsCursor::new(&module, instructions.clone());
        while cursor.current_instruction() != Some(WatInstruction::Call("$foo".to_string())) {
            cursor.next();
        }

        let new_instructions = vec![WatInstruction::I32Const(42), WatInstruction::I32Const(43)];

        let old_instructions = cursor
            .replace_current_call_with_arguments(new_instructions)
            .unwrap();

        // Check that we got back the original instructions
        assert_eq!(old_instructions.len(), 5);
        assert_eq!(old_instructions[0], WatInstruction::I32Const(1));
        assert_eq!(old_instructions[1], WatInstruction::I32Const(2));
        assert_eq!(old_instructions[2], WatInstruction::I32Add);
        assert_eq!(old_instructions[3], WatInstruction::I32Const(10));
        assert_eq!(
            old_instructions[4],
            WatInstruction::Call("$foo".to_string())
        );

        let instructions = instructions.borrow();
        // Check that the instructions were replaced correctly
        assert_eq!(instructions[0], WatInstruction::I32Const(999));
        assert_eq!(instructions[1], WatInstruction::I32Const(42));
        assert_eq!(instructions[2], WatInstruction::I32Const(43));
        assert_eq!(instructions[3], WatInstruction::I32Const(888));
    }

    #[test]
    fn test_replace_current() {
        let module = create_test_module();
        let instructions = Rc::new(RefCell::new(vec![
            WatInstruction::LocalGet("x".to_string()),
            WatInstruction::I32Const(1),
            WatInstruction::I32Add,
            WatInstruction::LocalGet("y".to_string()),
            WatInstruction::I32Mul,
            WatInstruction::Call("foo".to_string()),
        ]));

        {
            let mut cursor = InstructionsCursor::new(&module, instructions.clone());
            while cursor.current_instruction() != Some(WatInstruction::LocalGet("y".to_string())) {
                cursor.next();
            }

            cursor.replace_current(vec![
                WatInstruction::LocalGet("y".to_string()),
                WatInstruction::I32Const(1),
                WatInstruction::I32Add,
            ]);
        }

        let instructions = instructions.borrow();
        assert_eq!(instructions[0], WatInstruction::LocalGet("x".to_string()));
        assert_eq!(instructions[1], WatInstruction::I32Const(1));
        assert_eq!(instructions[2], WatInstruction::I32Add);
        assert_eq!(instructions[3], WatInstruction::LocalGet("y".to_string()));
        assert_eq!(instructions[4], WatInstruction::I32Const(1));
        assert_eq!(instructions[5], WatInstruction::I32Add);
        assert_eq!(instructions[6], WatInstruction::I32Mul);
    }

    #[test]
    fn test_insert_after_current() {
        let module = create_test_module();
        let instructions = Rc::new(RefCell::new(vec![
            WatInstruction::LocalGet("x".to_string()),
            WatInstruction::I32Const(1),
            WatInstruction::I32Add,
        ]));

        {
            let mut cursor = InstructionsCursor::new(&module, instructions.clone());
            // Move to I32Const(1)
            cursor.next();

            cursor.insert_after_current(vec![
                WatInstruction::LocalGet("y".to_string()),
                WatInstruction::I32Mul,
            ]);
        }

        let instructions = instructions.borrow();
        assert_eq!(instructions[0], WatInstruction::LocalGet("x".to_string()));
        assert_eq!(instructions[1], WatInstruction::I32Const(1));
        assert_eq!(instructions[2], WatInstruction::LocalGet("y".to_string()));
        assert_eq!(instructions[3], WatInstruction::I32Mul);
        assert_eq!(instructions[4], WatInstruction::I32Add);
    }

    #[test]
    fn test_try_catch_blocks() {
        let module = create_test_module();
        let try_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(1)]));
        let catch1_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(2)]));
        let catch2_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(3)]));
        let catch_all_block = vec![WatInstruction::I32Const(4)];

        let catches = vec![
            ("error1".to_string(), catch1_block.clone()),
            ("error2".to_string(), catch2_block.clone()),
        ];

        let instructions = Rc::new(RefCell::new(vec![WatInstruction::Try {
            try_block: try_block.clone(),
            catches: catches.clone(),
            catch_all: Some(catch_all_block.clone()),
        }]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let block_cursors = cursor.enter_block();

        // Should return 4 cursors: try block, 2 catch blocks, and catch_all block
        assert_eq!(block_cursors.len(), 4);

        // Check try block
        assert_eq!(
            block_cursors[0].current_instruction(),
            Some(WatInstruction::I32Const(1))
        );

        // Check first catch block
        assert_eq!(
            block_cursors[1].current_instruction(),
            Some(WatInstruction::I32Const(2))
        );

        // Check second catch block
        assert_eq!(
            block_cursors[2].current_instruction(),
            Some(WatInstruction::I32Const(3))
        );

        // Check catch_all block
        assert_eq!(
            block_cursors[3].current_instruction(),
            Some(WatInstruction::I32Const(4))
        );
    }

    #[test]
    fn test_standalone_catch_block() {
        let module = create_test_module();
        let catch_instructions = Rc::new(RefCell::new(vec![
            WatInstruction::I32Const(1),
            WatInstruction::I32Const(2),
        ]));

        let instructions = Rc::new(RefCell::new(vec![WatInstruction::Catch(
            "error".to_string(),
            catch_instructions.clone(),
        )]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let mut block_cursors = cursor.enter_block();

        assert_eq!(block_cursors.len(), 1);
        assert_eq!(
            block_cursors[0].current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
        assert!(block_cursors[0].next());
        assert_eq!(
            block_cursors[0].current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
    }

    #[test]
    fn test_standalone_catch_all_block() {
        let module = create_test_module();
        let catch_all_instructions = Rc::new(RefCell::new(vec![
            WatInstruction::I32Const(1),
            WatInstruction::Drop,
        ]));

        let instructions = Rc::new(RefCell::new(vec![WatInstruction::CatchAll(
            catch_all_instructions.clone(),
        )]));

        let cursor = InstructionsCursor::new(&module, instructions);
        let mut block_cursors = cursor.enter_block();

        assert_eq!(block_cursors.len(), 1);
        assert_eq!(
            block_cursors[0].current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
        assert!(block_cursors[0].next());
        assert_eq!(
            block_cursors[0].current_instruction(),
            Some(WatInstruction::Drop)
        );
    }
}
