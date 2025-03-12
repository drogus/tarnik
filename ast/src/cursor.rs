use crate::{
    FunctionKey, InstructionsList, InstructionsListWrapped, TypeDefinition, VecDebug, WasmType,
    WatFunction, WatInstruction, WatModule,
};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct StackElement {
    instructions: InstructionsListWrapped,
    position: usize,
    started: bool,
    block_arm: Option<BlockArm>,
}

impl StackElement {
    pub fn new(instructions: InstructionsListWrapped, block_arm: Option<BlockArm>) -> Self {
        Self {
            instructions,
            position: 0,
            started: false,
            block_arm,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockArmType {
    Try,
    Catch(String),
    CatchAll,
    If,
    Else,
    Block,
    Loop,
}

#[derive(Debug, Clone)]
pub struct BlockArm {
    pub instructions: InstructionsListWrapped,
    pub arm_type: BlockArmType,
    pub parent_instruction: Option<WatInstruction>,
}

#[derive(Debug)]
pub struct BlockArmsIterator {
    arms: VecDeque<BlockArm>,
    started: bool,
}

impl BlockArmsIterator {
    fn new(arms: VecDeque<BlockArm>) -> Self {
        Self {
            arms,
            started: false,
        }
    }

    pub fn next(&mut self, cursor: &mut InstructionsCursor) -> bool {
        if let Some(arm) = self.arms.pop_front() {
            if self.started {
                // Replace the current stack element
                cursor.stack.pop();
                cursor
                    .stack
                    .push(StackElement::new(arm.instructions.clone(), Some(arm)));
            } else {
                // First time through, just push
                cursor
                    .stack
                    .push(StackElement::new(arm.instructions.clone(), Some(arm)));
                self.started = true;
            }
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct InstructionsCursor<'a> {
    module: &'a mut WatModule,
    current_function: Option<FunctionKey>,
    stack: Vec<StackElement>,
}

impl<'a> Iterator for InstructionsCursor<'a> {
    type Item = WatInstruction;

    fn next(&mut self) -> Option<Self::Item> {
        // if we're just starting we want to set the current position
        // at the same time I haven't designed the code in the best way, so position is usize, so
        // initial position is already at 0. so here I'm doing a temporary hack to only advance
        // position if we haven't started yet
        // TODO: improve this
        if self.stack.last().unwrap().started {
            if self.current_position() + 1 < self.current_instructions_len() {
                self.advance_position(1);
                self.current_instruction()
            } else {
                None
            }
        } else {
            self.stack.last_mut().unwrap().started = true;
            self.current_instruction()
        }
    }
}

impl<'a> InstructionsCursor<'a> {
    pub fn new(module: &'a mut WatModule) -> Self {
        Self {
            module,
            stack: Default::default(),
            current_function: None,
        }
    }

    pub fn module(&self) -> &WatModule {
        self.module
    }

    pub fn module_mut(&mut self) -> &mut WatModule {
        self.module
    }

    pub fn current_function(&self) -> &WatFunction {
        self.get_function_by_key_unchecked(self.current_function.unwrap())
    }

    pub fn current_function_key(&self) -> FunctionKey {
        self.current_function.unwrap()
    }

    pub fn current_function_mut(&mut self) -> &mut WatFunction {
        self.get_function_by_key_unchecked_mut(self.current_function.unwrap())
    }

    pub fn get_function_by_key_unchecked(&self, key: FunctionKey) -> &WatFunction {
        self.module.get_function_by_key_unchecked(key)
    }

    pub fn get_function_by_key_unchecked_mut(&mut self, key: FunctionKey) -> &mut WatFunction {
        self.module.get_function_by_key_unchecked_mut(key)
    }

    pub fn add_function(&mut self, function: WatFunction) {
        self.module.add_function(function);
    }

    pub fn functions(&self) -> Vec<&WatFunction> {
        self.module.functions()
    }

    // TODO: this is not a great imepleemntation. we get a function by key and pass it to
    // `set_current_function`, which gets a key for a given name
    pub fn set_current_function_by_key(&mut self, key: FunctionKey) -> anyhow::Result<()> {
        let name = self.get_function_by_key_unchecked(key).name.clone();
        self.set_current_function(&name)
    }

    pub fn set_current_function(&mut self, name: &str) -> anyhow::Result<()> {
        let key = self
            .module
            .get_function_key(name)
            .ok_or(anyhow::anyhow!("Could not find function {name}"))?;

        let function = self.module.get_function_by_key_unchecked(key);
        self.current_function = Some(key);
        self.stack = vec![];
        self.stack
            .push(StackElement::new(function.body.clone(), None));

        Ok(())
    }

    pub fn current_position(&self) -> usize {
        // I'm not sure about the unwraps here. For now I just assume we will always have at
        // least one element on the stack (the function's body), but I haven't thought about it too
        // much, so it might be good to think it through at some point
        self.stack.last().unwrap().position
    }

    pub fn advance_position(&mut self, steps: usize) {
        self.stack.last_mut().unwrap().position += steps;
    }

    pub fn retract_position(&mut self, steps: usize) {
        // TODO: check if we are not ending up < 0
        self.stack.last_mut().unwrap().position -= steps;
    }

    pub fn current_instructions_list(&self) -> InstructionsListWrapped {
        self.stack.last().unwrap().instructions.clone()
    }

    pub fn replace_current(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<WatInstruction> {
        let position = self.current_position();
        let instructions_list = self.current_instructions_list();
        let mut instructions = instructions_list.borrow_mut();
        if position < instructions.len() {
            let old = instructions[position].clone();

            // Replace the current instruction with the first new one
            instructions[position] = new_instructions[0].clone();

            // Insert any remaining instructions after the current position
            if new_instructions.len() > 1 {
                let remaining = &new_instructions[1..];
                instructions.splice(position + 1..position + 1, remaining.iter().cloned());
            }

            self.advance_position(new_instructions.len() - 1);

            Some(old)
        } else {
            None
        }
    }

    pub fn insert_after_current(&mut self, new_instructions: Vec<WatInstruction>) {
        let mut position = self.current_position();
        let instructions_list = self.current_instructions_list();
        let mut instructions = instructions_list.borrow_mut();
        if position >= instructions.len() {
            position = instructions.len() - 1;
            self.set_position(position);
        }
        // Insert all new instructions after the current position
        instructions.splice(position + 1..position + 1, new_instructions.iter().cloned());
    }

    pub fn current_instruction(&self) -> Option<WatInstruction> {
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();
        if self.current_position() < instructions.len() {
            Some(instructions[self.current_position()].clone())
        } else {
            None
        }
    }

    pub fn current_instructions_len(&self) -> usize {
        self.stack.last().unwrap().instructions.borrow().len()
    }

    pub fn is_last(&self) -> bool {
        self.current_position() == self.current_instructions_len() - 1
    }

    pub fn previous(&mut self) -> Option<WatInstruction> {
        if self.current_position() > 0 {
            self.retract_position(1);
            self.current_instruction()
        } else {
            None
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
                    .functions()
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
                    .functions()
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
                    .functions()
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
            WatInstruction::RefEq => (2, 1),
        }
    }

    // TODO: can I somehow return Vec<Vec<&WatInstruction>> here?
    pub fn get_arguments_instructions(&'a self, count: usize) -> Vec<Vec<WatInstruction>> {
        let mut result = Vec::new();
        let mut stack_size = 0isize;
        // TODO: get rid of these unwraps
        let mut pos = self.current_position();
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();

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

    pub fn enter_block(&mut self) -> anyhow::Result<BlockArmsIterator> {
        if let Some(current) = self.current_instruction() {
            let mut arms = VecDeque::new();
            let current_clone = current.clone();

            match current {
                WatInstruction::Block { instructions, .. } => {
                    arms.push_back(BlockArm {
                        instructions,
                        arm_type: BlockArmType::Block,
                        parent_instruction: Some(current_clone),
                    });
                }
                WatInstruction::Loop { instructions, .. } => {
                    arms.push_back(BlockArm {
                        instructions,
                        arm_type: BlockArmType::Loop,
                        parent_instruction: Some(current_clone),
                    });
                }
                WatInstruction::If { then, r#else } => {
                    arms.push_back(BlockArm {
                        instructions: then,
                        arm_type: BlockArmType::If,
                        parent_instruction: Some(current_clone.clone()),
                    });
                    if let Some(e) = r#else {
                        arms.push_back(BlockArm {
                            instructions: e,
                            arm_type: BlockArmType::Else,
                            parent_instruction: Some(current_clone),
                        });
                    }
                }
                WatInstruction::Try {
                    try_block,
                    catches,
                    catch_all,
                } => {
                    arms.push_back(BlockArm {
                        instructions: try_block,
                        arm_type: BlockArmType::Try,
                        parent_instruction: Some(current_clone.clone()),
                    });

                    for (tag, instructions) in catches {
                        arms.push_back(BlockArm {
                            instructions,
                            arm_type: BlockArmType::Catch(tag.clone()),
                            parent_instruction: Some(current_clone.clone()),
                        });
                    }

                    if let Some(c) = catch_all {
                        arms.push_back(BlockArm {
                            instructions: c,
                            arm_type: BlockArmType::CatchAll,
                            parent_instruction: Some(current_clone),
                        });
                    }
                }
                _ => {}
            }

            Ok(BlockArmsIterator::new(arms))
        } else {
            Err(anyhow::anyhow!("Not a block instruction"))
        }
    }
    pub fn stack_level(&self) -> usize {
        self.stack.len()
    }

    pub fn exit_block(&mut self) {
        if self.stack.len() == 1 {
            panic!();
        }
        // TODO: this should panic if we try to remove current function's instructions
        self.stack.pop();
    }

    pub fn earliest_argument(&self) -> Option<usize> {
        self.get_call_arguments().map(|args| {
            self.current_position()
                - args
                    .into_iter()
                    .flatten()
                    .collect::<InstructionsList>()
                    .len()
        })
    }

    pub fn get_call_arguments(&self) -> Option<Vec<Vec<WatInstruction>>> {
        match self.current_instruction() {
            Some(instr) => {
                let args = self.instruction_stack_effect(&instr).0;
                Some(self.get_arguments_instructions(args))
            }
            _ => None,
        }
    }

    pub fn is_in_top_level_block(&self) -> bool {
        self.stack.len() == 1
    }

    fn last_instruction(&self) -> bool {
        self.current_position() + 1 == self.current_instructions_list().borrow().len()
    }

    // TODO: at the moment this assumes that there are no other block types than Try, which is
    // true for Jawsm (as all the blocks and loops are replaced first), but it won't be true
    // in general
    pub fn get_parent_try_catch_blocks(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let mut current_instructions: Vec<WatInstruction> = Vec::new();
        current_instructions.extend(new_instructions);

        if let Some(stack_element) = self.stack.last() {
            if let Some(block_arm) = &stack_element.block_arm {
                if let BlockArmType::Try = block_arm.arm_type {
                    match &block_arm.parent_instruction {
                            Some(WatInstruction::Try { try_block: _, catches, catch_all }) => {
                                return Some(vec![WatInstruction::Try { try_block: Rc::new(RefCell::new(current_instructions)), catches: catches.clone(), catch_all: catch_all.clone() }]);
                                },
                            _ => unreachable!("if BlockArmType is set to Try, the parent instruction should also be a try")
                        };
                }
            }
        }

        Some(current_instructions)
    }

    pub fn fetch_till_the_end_of_function_try_catch_aware(
        &mut self,
        new_instructions: Option<Vec<WatInstruction>>,
    ) -> Option<Vec<WatInstruction>> {
        let mut saved_stack = Vec::new();

        let pos = self.current_position();
        // Replace current block first
        let mut all_removed =
            self.fetch_till_the_end_of_the_block_try_catch_aware(new_instructions, vec![])?;

        // Save and exit blocks while processing
        while !self.is_in_top_level_block() {
            saved_stack.push(self.stack.last().unwrap().clone());
            self.exit_block();

            if !self.last_instruction() {
                all_removed =
                    self.fetch_till_the_end_of_the_block_try_catch_aware(None, all_removed)?;
            }
        }

        // Restore the stack in reverse order
        for el in saved_stack.into_iter().rev() {
            self.stack.push(el);
        }

        Some(all_removed)
    }

    pub fn replace_till_the_end_of_function_try_catch_aware(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let mut saved_stack = Vec::new();

        let pos = self.current_position();
        // Replace current block first
        let mut all_removed =
            self.replace_till_the_end_of_the_block_try_catch_aware(new_instructions, vec![])?;

        // Save and exit blocks while processing
        while !self.is_in_top_level_block() {
            saved_stack.push(self.stack.last().unwrap().clone());
            self.exit_block();

            if !self.last_instruction() {
                all_removed =
                    self.fetch_till_the_end_of_the_block_try_catch_aware(None, all_removed)?;
            }
        }

        // Restore the stack in reverse order
        for el in saved_stack.into_iter().rev() {
            self.stack.push(el);
        }

        Some(all_removed)
    }

    pub fn replace_till_the_end_of_function(
        &mut self,
        new_instructions: Vec<WatInstruction>,
        append_instructions: Option<Vec<WatInstruction>>,
    ) -> Option<Vec<WatInstruction>> {
        let mut all_removed = Vec::new();
        let mut saved_stack = Vec::new();

        let _pos = self.current_position();
        // Replace current block first
        let removed = self.replace_till_the_end_of_the_block(new_instructions)?;
        all_removed.extend(removed);

        // Save and exit blocks while processing
        while !self.is_in_top_level_block() {
            saved_stack.push(self.stack.last().unwrap().clone());
            self.exit_block();

            if !self.last_instruction() {
                self.next();
                let removed = self.replace_till_the_end_of_the_block(vec![])?;
                all_removed.extend(removed);
            }
        }

        // Restore the stack in reverse order
        for el in saved_stack.into_iter().rev() {
            self.stack.push(el);
        }

        // Append additional instructions if provided
        if let Some(instructions) = append_instructions {
            let function_body = self.current_function().body.clone();
            function_body.borrow_mut().extend(instructions);
        }

        Some(all_removed)
    }

    pub fn replace_till_the_end_of_the_block(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let current_pos = self.current_position();
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();
        let end_pos = if instructions.is_empty() {
            0
        } else {
            instructions.len() - 1
        };

        drop(instructions); // Release the borrow before calling replace_range
        self.replace_range(current_pos, end_pos, new_instructions)
    }

    pub fn fetch_till_the_end_of_the_block_try_catch_aware(
        &mut self,
        new_instructions: Option<Vec<WatInstruction>>,
        mut current_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let current_pos = self.current_position() + 1;
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();
        let end_pos = if instructions.is_empty() {
            0
        } else {
            instructions.len() - 1
        };

        drop(instructions); // Release the borrow before calling replace_range
        let mut fetched_instructions =
            new_instructions.or_else(|| self.fetch_range(current_pos, end_pos));

        if let Some(mut instructions) = fetched_instructions {
            current_instructions.extend(instructions);

            if let Some(stack_element) = self.stack.last() {
                if let Some(block_arm) = &stack_element.block_arm {
                    if let BlockArmType::Try = block_arm.arm_type {
                        match &block_arm.parent_instruction {
                            Some(WatInstruction::Try { try_block: _, catches, catch_all }) => {
                                return Some(vec![WatInstruction::Try { try_block: Rc::new(RefCell::new(current_instructions)), catches: catches.clone(), catch_all: catch_all.clone() }]);
                                },
                            _ => unreachable!("if BlockArmType is set to Try, the parent instruction should also be a try")
                        };
                    }
                }
            }
        }

        Some(current_instructions)
    }

    pub fn replace_till_the_end_of_the_block_try_catch_aware(
        &mut self,
        new_instructions: Vec<WatInstruction>,
        mut current_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let current_pos = self.current_position();
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();
        let end_pos = if instructions.is_empty() {
            0
        } else {
            instructions.len() - 1
        };

        drop(instructions); // Release the borrow before calling replace_range
        let mut removed_instructions = self.replace_range(current_pos, end_pos, new_instructions);

        if let Some(mut removed) = removed_instructions {
            current_instructions.extend(removed);

            if let Some(stack_element) = self.stack.last() {
                if let Some(block_arm) = &stack_element.block_arm {
                    if let BlockArmType::Try = block_arm.arm_type {
                        match &block_arm.parent_instruction {
                            Some(WatInstruction::Try { try_block: _, catches, catch_all }) => {
                                return Some(vec![WatInstruction::Try { try_block: Rc::new(RefCell::new(current_instructions)), catches: catches.clone(), catch_all: catch_all.clone() }]);
                                },
                            _ => unreachable!("if BlockArmType is set to Try, the parent instruction should also be a try")
                        };
                    }
                }
            }
        }

        Some(current_instructions)
    }

    pub fn fetch_range(&mut self, start: usize, end: usize) -> Option<Vec<WatInstruction>> {
        let instructions_list = self.current_instructions_list();
        let mut instructions = instructions_list.borrow_mut();
        let len = instructions.len();
        if start > end || end >= len {
            return None;
        }

        Some(instructions[start..=end].iter().cloned().collect())
    }

    pub fn replace_range(
        &mut self,
        start: usize,
        end: usize,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        let instructions_list = self.current_instructions_list();
        let mut instructions = instructions_list.borrow_mut();
        let len = instructions.len();
        if start > end || end >= len {
            return None;
        }

        let mut old_instructions = Vec::new();
        {
            old_instructions.extend(instructions[start..=end].iter().cloned());
        }

        instructions.splice(start..=end, new_instructions);

        Some(old_instructions)
    }

    pub fn reset(&mut self) {
        self.set_position(0);
        self.stack.last_mut().unwrap().started = false;
    }

    pub fn set_position(&mut self, position: usize) {
        self.stack.last_mut().unwrap().position = position;
    }

    pub fn replace_current_call_with_arguments(
        &mut self,
        new_instructions: Vec<WatInstruction>,
    ) -> Option<Vec<WatInstruction>> {
        match self.current_instruction() {
            Some(WatInstruction::Call(name)) => {
                if let Some(func) = self
                    .module
                    .functions()
                    .iter()
                    .find(|f| format!("${}", f.name) == name)
                {
                    let param_count = func.params.len();
                    let mut old_instructions = Vec::new();
                    let mut stack_size = 0isize;
                    let mut pos = self.current_position();
                    let instructions_list = self.current_instructions_list();
                    let mut instructions = instructions_list.borrow_mut();
                    let original_position = pos;

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
                    self.set_position(pos + new_instructions_len - 1);

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

#[derive(Debug, Default)]
pub struct StackState {
    types: VecDeque<WasmType>,
}

impl StackState {
    pub fn new() -> Self {
        Self {
            types: VecDeque::new(),
        }
    }

    pub fn push(&mut self, ty: WasmType) {
        self.types.push_back(ty);
    }

    pub fn pop_front(&mut self) -> Option<WasmType> {
        self.types.pop_front()
    }

    pub fn pop(&mut self) -> Option<WasmType> {
        self.types.pop_back()
    }

    pub fn peek(&self) -> Option<&WasmType> {
        self.types.back()
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn clear(&mut self) {
        self.types.clear();
    }
}

impl InstructionsCursor<'_> {
    pub fn analyze_stack_state_until_current(&self) -> StackState {
        let instructions_list = self.current_instructions_list();
        let instructions = instructions_list.borrow();
        let current_pos = self.current_position();
        self.analyze_stack_state(&instructions[..current_pos])
    }

    pub fn analyze_stack_state_for_arguments(&self) -> Option<StackState> {
        let current = self.current_instruction()?;
        let args_count = self.instruction_stack_effect(&current).0;
        let args_instructions = self
            .get_arguments_instructions(args_count)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Some(self.analyze_stack_state(&args_instructions))
    }

    pub fn analyze_stack_state(&self, instructions: &[WatInstruction]) -> StackState {
        let mut state = StackState::new();

        for instruction in instructions {
            match instruction {
                WatInstruction::I32Const(_) => state.push(WasmType::I32),
                WatInstruction::I64Const(_) => state.push(WasmType::I64),
                WatInstruction::F32Const(_) => state.push(WasmType::F32),
                WatInstruction::F64Const(_) => state.push(WasmType::F64),

                WatInstruction::I32Add
                | WatInstruction::I32Sub
                | WatInstruction::I32Mul
                | WatInstruction::I32DivS
                | WatInstruction::I32DivU => {
                    state.pop(); // Pop two I32s
                    state.pop();
                    state.push(WasmType::I32); // Push result
                }

                WatInstruction::I64Add
                | WatInstruction::I64Sub
                | WatInstruction::I64Mul
                | WatInstruction::I64DivS
                | WatInstruction::I64DivU => {
                    state.pop(); // Pop two I64s
                    state.pop();
                    state.push(WasmType::I64); // Push result
                }

                WatInstruction::LocalGet(name) => {
                    if let Some(ty) = self.current_function().locals.get(name) {
                        state.push(ty.clone());
                    }
                }

                WatInstruction::LocalSet(_name) => {
                    state.pop(); // Pop the value being set
                }

                WatInstruction::Call(name) => {
                    if let Some(func) = self.module.get_function(name) {
                        // Pop arguments
                        for _ in &func.params {
                            state.pop();
                        }
                        // Push result if any
                        if let Some(result_type) = func.results.first() {
                            state.push(result_type.clone());
                        }
                    }
                }

                WatInstruction::Drop => {
                    state.pop();
                }
                WatInstruction::Nop => (),
                WatInstruction::Local { name: _, ty: _ } => (),
                WatInstruction::GlobalGet(name) => {
                    if let Some(global) = self.module.globals.get(name) {
                        state.push(global.ty.clone());
                    }
                }
                WatInstruction::GlobalSet(_name) => {
                    state.pop(); // Remove the value being set
                }
                WatInstruction::CallRef(name) => {
                    if let Some(func) = self.module.get_function(name) {
                        // Pop arguments
                        for _ in &func.params {
                            state.pop();
                        }
                        // Push result if any
                        if let Some(result_type) = func.results.first() {
                            state.push(result_type.clone());
                        }
                    }
                }
                WatInstruction::F32Neg | WatInstruction::F64Neg => {
                    state.pop();
                    state.push(WasmType::F32);
                }
                WatInstruction::I32Eqz
                | WatInstruction::I64Eqz
                | WatInstruction::F32Eqz
                | WatInstruction::F64Eqz => {
                    state.pop();
                    state.push(WasmType::I32); // Comparison results are i32
                }
                WatInstruction::I32Eq
                | WatInstruction::I64Eq
                | WatInstruction::F32Eq
                | WatInstruction::F64Eq
                | WatInstruction::I32Ne
                | WatInstruction::I64Ne
                | WatInstruction::F32Ne
                | WatInstruction::F64Ne => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::F32Add
                | WatInstruction::F64Add
                | WatInstruction::F32Sub
                | WatInstruction::F64Sub
                | WatInstruction::F32Mul
                | WatInstruction::F64Mul
                | WatInstruction::F32Div
                | WatInstruction::F64Div => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::F32);
                }
                WatInstruction::I32RemS
                | WatInstruction::I64RemS
                | WatInstruction::I32RemU
                | WatInstruction::I64RemU => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I32And
                | WatInstruction::I64And
                | WatInstruction::I32Or
                | WatInstruction::I64Or
                | WatInstruction::I32Xor
                | WatInstruction::I64Xor => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I32LtS
                | WatInstruction::I64LtS
                | WatInstruction::I32LtU
                | WatInstruction::I64LtU
                | WatInstruction::F32Lt
                | WatInstruction::F64Lt
                | WatInstruction::I32LeS
                | WatInstruction::I64LeS
                | WatInstruction::I32LeU
                | WatInstruction::I64LeU
                | WatInstruction::F32Le
                | WatInstruction::F64Le
                | WatInstruction::I32GeS
                | WatInstruction::I64GeS
                | WatInstruction::I32GeU
                | WatInstruction::I64GeU
                | WatInstruction::F32Ge
                | WatInstruction::F64Ge
                | WatInstruction::I32GtS
                | WatInstruction::I64GtS
                | WatInstruction::I32GtU
                | WatInstruction::I64GtU
                | WatInstruction::F32Gt
                | WatInstruction::F64Gt => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I32Shl
                | WatInstruction::I64Shl
                | WatInstruction::I32ShrS
                | WatInstruction::I64ShrS
                | WatInstruction::I32ShrU
                | WatInstruction::I64ShrU => {
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I64ExtendI32S => {
                    state.pop();
                    state.push(WasmType::I64);
                }
                WatInstruction::I32WrapI64 => {
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::F64PromoteF32 => {
                    state.pop();
                    state.push(WasmType::F64);
                }
                WatInstruction::F32DemoteF64 => {
                    state.pop();
                    state.push(WasmType::F32);
                }
                WatInstruction::F32ConvertI32S
                | WatInstruction::F32ConvertI32U
                | WatInstruction::F32ConvertI64S
                | WatInstruction::F32ConvertI64U => {
                    state.pop();
                    state.push(WasmType::F32);
                }
                WatInstruction::F64ConvertI32S
                | WatInstruction::F64ConvertI32U
                | WatInstruction::F64ConvertI64S
                | WatInstruction::F64ConvertI64U => {
                    state.pop();
                    state.push(WasmType::F64);
                }
                WatInstruction::I32TruncF32S
                | WatInstruction::I32TruncF32U
                | WatInstruction::I32TruncF64S
                | WatInstruction::I32TruncF64U => {
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I64TruncF32S
                | WatInstruction::I64TruncF32U
                | WatInstruction::I64TruncF64S
                | WatInstruction::I64TruncF64U => {
                    state.pop();
                    state.push(WasmType::I64);
                }
                WatInstruction::I32ReinterpretF32 => {
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::F32ReinterpretI32 => {
                    state.pop();
                    state.push(WasmType::F32);
                }
                WatInstruction::I64ReinterpretF64 => {
                    state.pop();
                    state.push(WasmType::I64);
                }
                WatInstruction::F64ReinterpretI64 => {
                    state.pop();
                    state.push(WasmType::F64);
                }
                WatInstruction::I31GetS | WatInstruction::I31GetU => {
                    state.pop();
                    state.push(WasmType::I32);
                }
                WatInstruction::I32Store(_)
                | WatInstruction::I64Store(_)
                | WatInstruction::F32Store(_)
                | WatInstruction::F64Store(_)
                | WatInstruction::I32Store8(_)
                | WatInstruction::I32Store16(_)
                | WatInstruction::I64Store8(_)
                | WatInstruction::I64Store16(_)
                | WatInstruction::I64Store32(_) => {
                    state.pop(); // Value to store
                    state.pop(); // Address
                }
                WatInstruction::I32Load(_)
                | WatInstruction::I64Load(_)
                | WatInstruction::F32Load(_)
                | WatInstruction::F64Load(_)
                | WatInstruction::I32Load8S(_)
                | WatInstruction::I32Load8U(_)
                | WatInstruction::I32Load16S(_)
                | WatInstruction::I32Load16U(_)
                | WatInstruction::I64Load8S(_)
                | WatInstruction::I64Load8U(_)
                | WatInstruction::I64Load16S(_)
                | WatInstruction::I64Load16U(_)
                | WatInstruction::I64Load32S(_)
                | WatInstruction::I64Load32U(_) => {
                    state.pop(); // Address
                    state.push(match instruction {
                        WatInstruction::I32Load(_)
                        | WatInstruction::I32Load8S(_)
                        | WatInstruction::I32Load8U(_)
                        | WatInstruction::I32Load16S(_)
                        | WatInstruction::I32Load16U(_) => WasmType::I32,
                        WatInstruction::I64Load(_)
                        | WatInstruction::I64Load8S(_)
                        | WatInstruction::I64Load8U(_)
                        | WatInstruction::I64Load16S(_)
                        | WatInstruction::I64Load16U(_)
                        | WatInstruction::I64Load32S(_)
                        | WatInstruction::I64Load32U(_) => WasmType::I64,
                        WatInstruction::F32Load(_) => WasmType::F32,
                        WatInstruction::F64Load(_) => WasmType::F64,
                        _ => unreachable!(),
                    });
                }
                WatInstruction::StructNew(name) => {
                    if let Some(ty) = self.module.get_type_by_name(name) {
                        if let WasmType::Struct(fields) = ty {
                            for _ in fields {
                                state.pop(); // Pop field values
                            }
                        }
                        state.push(WasmType::r#ref(name.clone()));
                    }
                }
                WatInstruction::StructGet(struct_type, _) => {
                    state.pop(); // Pop struct reference
                    if let Some(ty) = self.module.get_type_by_name(struct_type) {
                        if let WasmType::Struct(fields) = ty {
                            if let Some(field) = fields.first() {
                                state.push(field.ty.clone());
                            }
                        }
                    }
                }
                WatInstruction::StructSet(_, _) => {
                    state.pop(); // Value to set
                    state.pop(); // Struct reference
                }
                WatInstruction::ArrayNew(_) => {
                    state.pop(); // Initial value
                    state.pop(); // Length
                    state.push(WasmType::I32); // Array reference
                }
                WatInstruction::ArrayNewFixed(_, count) => {
                    for _ in 0..*count {
                        state.pop(); // Initial values
                    }
                    state.push(WasmType::I32); // Array reference
                }
                WatInstruction::ArrayLen => {
                    state.pop(); // Array reference
                    state.push(WasmType::I32);
                }
                WatInstruction::ArrayGet(_) | WatInstruction::ArrayGetU(_) => {
                    state.pop(); // Index
                    state.pop(); // Array reference
                    state.push(WasmType::I32);
                }
                WatInstruction::ArraySet(_) => {
                    state.pop(); // Value
                    state.pop(); // Index
                    state.pop(); // Array reference
                }
                WatInstruction::RefNull(ty) => {
                    state.push(ty.clone());
                }
                WatInstruction::RefCast(ty) | WatInstruction::RefTest(ty) => {
                    state.pop();
                    state.push(ty.clone());
                }
                WatInstruction::Ref(_) => {
                    state.push(WasmType::I32);
                }
                WatInstruction::RefFunc(_) => {
                    state.push(WasmType::I32);
                }
                WatInstruction::Type(_) => (),
                WatInstruction::Return => {
                    if let Some(result_type) = self.current_function().results.first() {
                        state.pop(); // Pop return value
                        state.push(result_type.clone());
                    }
                }
                WatInstruction::ReturnCall(name) => {
                    if let Some(func) = self.module.get_function(name) {
                        for _ in &func.params {
                            state.pop();
                        }
                        if let Some(result_type) = func.results.first() {
                            state.push(result_type.clone());
                        }
                    }
                }
                WatInstruction::Block { signature, .. } => {
                    // Handle block parameters and results
                    for _ in &signature.params {
                        state.pop();
                    }
                    if let Some(result_type) = &signature.result {
                        state.push(result_type.clone());
                    }
                }
                WatInstruction::Loop { instructions, .. } => {
                    // Loop instructions are handled recursively
                    let loop_state = self.analyze_stack_state(&instructions.borrow());
                    if let Some(ty) = loop_state.peek() {
                        state.push(ty.clone());
                    }
                }
                WatInstruction::If { then, r#else } => {
                    state.pop(); // Condition
                    let then_state = self.analyze_stack_state(&then.borrow());
                    if let Some(r#else) = r#else {
                        let _else_state = self.analyze_stack_state(&r#else.borrow());
                        // If both branches produce a value, use the then branch type
                        if let Some(ty) = then_state.peek() {
                            state.push(ty.clone());
                        }
                    }
                }
                WatInstruction::BrIf(_) => {
                    state.pop(); // Condition
                }
                WatInstruction::Br(_) => (),
                WatInstruction::Empty => (),
                WatInstruction::Log => {
                    state.pop(); // Value to log
                }
                WatInstruction::Identifier(_) => (),
                WatInstruction::LocalTee(name) => {
                    if let Some(ty) = self.current_function().locals.get(name) {
                        state.pop(); // Pop the value
                        state.push(ty.clone()); // Push it back
                    }
                }
                WatInstruction::RefI31 => {
                    state.pop();
                    state.push(WasmType::I31Ref);
                }
                WatInstruction::Throw(_) => {
                    state.pop(); // Exception value
                }
                WatInstruction::Try {
                    try_block,
                    catches,
                    catch_all,
                } => {
                    let try_state = self.analyze_stack_state(&try_block.borrow());
                    if let Some(ty) = try_state.peek() {
                        state.push(ty.clone());
                    }
                }
                WatInstruction::RefEq => {
                    state.pop();
                    state.pop();
                    state.push(WasmType::I32);
                }
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::{test_helpers::*, wat_converter::parse_wat_function};

    #[test]
    fn test_replace_till_end_of_function() {
        let mut module = WatModule::new();

        // Input function in WAT format
        let input_wat = r#"
            (func $bar
                i32.const 1
                if
                    i32.const 1
                    if
                        i32.const 10
                        call $log
                        i32.const 44
                        i32.const 22
                        i32.add
                        call $log
                    else
                        i32.const 100
                        call $log
                    end
                    i32.const 10
                    call $foo
                else
                    i32.const 666
                    call $log
                end
                i32.const 50
                call $bar
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("bar").unwrap();

        // Navigate to i32.const 44 in the inner if
        cursor.next(); // i32.const 1
        cursor.next(); // if
        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        cursor.next(); // i32.const 1
        cursor.next(); // if
        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        cursor.next(); // i32.const 10
        cursor.next(); // call $log
        cursor.next(); // i32.const 44

        let replacement = vec![WatInstruction::Call("$the_replacement".to_string())];
        let append_instructions = Some(vec![
            WatInstruction::I32Const(777),
            WatInstruction::Call("$appended".to_string()),
        ]);

        cursor
            .replace_till_the_end_of_function(replacement, append_instructions)
            .unwrap();

        // Expected function after transformation
        let expected_wat = r#"
            (func $bar
                i32.const 1
                if
                  i32.const 1
                  if
                    i32.const 10
                    call $log
                    call $the_replacement
                  else
                    i32.const 100
                    call $log
                  end
                else
                  i32.const 666
                  call $log
                end
                i32.const 777
                call $appended
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("bar").unwrap();

        assert_functions_eq(&expected, actual);
    }

    #[test]
    fn test_stack_analysis_until_current() {
        let mut module = WatModule::new();

        let input_wat = r#"
            (func $test
                i32.const 10
                i32.const 20
                i32.add
                i64.const 30
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        // Test initial position (first instruction)
        cursor.next().unwrap();
        let state = cursor.analyze_stack_state_until_current();
        assert_eq!(state.len(), 0);

        cursor.next().unwrap();
        let state = cursor.analyze_stack_state_until_current();
        assert_eq!(state.len(), 1);
        assert_eq!(state.peek().unwrap(), &WasmType::I32);

        // Move to add instruction
        cursor.next();
        let state = cursor.analyze_stack_state_until_current();
        assert_eq!(state.len(), 2);
        let types: Vec<_> = state.types.iter().collect();
        assert_eq!(types[0], &WasmType::I32);
        assert_eq!(types[1], &WasmType::I32);

        // Move to final instruction
        cursor.next();
        let state = cursor.analyze_stack_state_until_current();
        assert_eq!(state.len(), 1);
        assert_eq!(state.peek().unwrap(), &WasmType::I32);
    }

    #[test]
    fn test_stack_analysis() {
        let mut module = WatModule::new();

        let input_wat = r#"
            (func $test
                i32.const 10
                i32.const 20
                i32.add
                i64.const 30
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        let state = cursor.analyze_stack_state(&cursor.current_instructions_list().borrow());
        assert_eq!(state.len(), 2);

        let types: Vec<_> = state.types.iter().collect();
        assert_eq!(types[0], &WasmType::I32);
        assert_eq!(types[1], &WasmType::I64);
    }

    #[test]
    fn test_stack_analysis_with_locals() {
        let mut module = WatModule::new();

        let input_wat = r#"
            (func $test
                (local $x i32)
                i32.const 10
                local.set $x
                local.get $x
                i32.const 20
                i32.add
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        let state = cursor.analyze_stack_state(&cursor.current_instructions_list().borrow());
        assert_eq!(state.len(), 1);
        assert_eq!(state.peek().unwrap(), &WasmType::I32);
    }

    fn create_test_module() -> WatModule {
        let mut module = WatModule::new();

        // Add a test function that takes 2 i32 parameters
        let mut func = WatFunction::new("foo");
        func.add_param("a", &WasmType::I32);
        func.add_param("b", &WasmType::I32);
        module.add_function(func);

        let func = WatFunction::new("bar");
        module.add_function(func);

        module
    }

    #[test]
    fn test_cursor_navigation() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                i32.const 1
                i32.const 2
                i32.add
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").ok();

        assert_eq!(cursor.next(), Some(WatInstruction::I32Const(1)));
        assert_eq!(cursor.next(), Some(WatInstruction::I32Const(2)));
        assert_eq!(cursor.next(), Some(WatInstruction::I32Add));
        assert_eq!(cursor.next(), None);

        assert_eq!(cursor.previous(), Some(WatInstruction::I32Const(2)));
    }

    #[test]
    fn test_block_navigation() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                block $test
                    i32.const 1
                    i32.const 2
                end
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);

        assert_eq!(cursor.next(), Some(WatInstruction::I32Const(1)));
        assert_eq!(cursor.next(), Some(WatInstruction::I32Const(2)));
    }

    #[test]
    fn test_function_arguments() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test (param $a i32) (param $b i32)
                i32.const 10
                i32.const 1
                i32.const 2
                i32.add
                call $foo
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        while cursor.next() != Some(WatInstruction::Call("$foo".to_string())) {}

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
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test (param $a i32) (param $b i32)
                (local $var i32)
                i32.const 10
                i32.const 1
                local.set $var
                local.get $var
                i32.const 2
                i32.add
                call $foo
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        while cursor.next() != Some(WatInstruction::Call("$foo".to_string())) {}

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
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                block $outer
                    block $inner
                        i32.const 1
                    end
                    i32.const 2
                end
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        cursor.next();
        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        cursor.next();

        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );
    }

    #[test]
    fn test_if_block() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                if
                    i32.const 1
                else
                    i32.const 2
                end
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);

        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );

        assert!(iterator.next(&mut cursor));
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );
    }

    #[test]
    fn test_replace_current_call_with_arguments() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                i32.const 999
                i32.const 1
                i32.const 2
                i32.add
                i32.const 10
                call $foo
                i32.const 888
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();
        while cursor.next() != Some(WatInstruction::Call("$foo".to_string())) {}

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

        // Check that the instructions were replaced correctly
        let expected_wat = r#"
            (func $test
                i32.const 999
                i32.const 42
                i32.const 43
                i32.const 888
            )"#;

        let expected_func = parse_wat_function(expected_wat);
        let actual_func = module.get_function("test").unwrap();

        assert_functions_eq(&expected_func, actual_func);
    }

    #[test]
    fn test_replace_current() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                local.get $x
                i32.const 1
                i32.add
                local.get $y
                i32.mul
                call $foo
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        {
            let mut cursor = InstructionsCursor::new(&mut module);
            cursor.set_current_function("test").unwrap();
            while cursor.next() != Some(WatInstruction::LocalGet("$y".to_string())) {}

            cursor.replace_current(vec![
                WatInstruction::LocalGet("$y".to_string()),
                WatInstruction::I32Const(1),
                WatInstruction::I32Add,
            ]);
        }

        let expected_wat = r#"
            (func $test
                local.get $x
                i32.const 1
                i32.add
                local.get $y
                i32.const 1
                i32.add
                i32.mul
                call $foo
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);
    }

    #[test]
    fn test_insert_after_current() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                local.get $x
                i32.const 1
                i32.add
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        {
            let mut cursor = InstructionsCursor::new(&mut module);
            cursor.set_current_function("test").unwrap();
            // Move to i32.const 1
            cursor.next();
            cursor.next();

            cursor.insert_after_current(vec![
                WatInstruction::LocalGet("$y".to_string()),
                WatInstruction::I32Mul,
            ]);
        }

        let expected_wat = r#"
            (func $test
                local.get $x
                i32.const 1
                local.get $y
                i32.mul
                i32.add
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);
    }

    #[test]
    fn test_try_catch_blocks() {
        let mut module = create_test_module();
        let try_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(1)]));
        let catch1_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(2)]));
        let catch2_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(3)]));
        let catch_all_block = Rc::new(RefCell::new(vec![WatInstruction::I32Const(4)]));

        let catches = vec![
            ("error1".to_string(), catch1_block.clone()),
            ("error2".to_string(), catch2_block.clone()),
        ];

        let instructions = vec![WatInstruction::Try {
            try_block: try_block.clone(),
            catches: catches.clone(),
            catch_all: Some(catch_all_block.clone()),
        }];

        module
            .get_function_mut("bar")
            .unwrap()
            .set_body(instructions);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("bar").unwrap();

        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(1))
        );

        iterator.next(&mut cursor);
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(2))
        );

        iterator.next(&mut cursor);
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(3))
        );

        iterator.next(&mut cursor);
        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::I32Const(4))
        );

        assert!(!iterator.next(&mut cursor));
    }

    #[test]
    fn test_replace_range() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                i32.const 1
                i32.const 2
                i32.add
                i32.const 3
                i32.mul
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();

        // Replace instructions from index 1 to 3 (inclusive)
        let old_instructions = cursor
            .replace_range(
                1,
                3,
                vec![WatInstruction::I32Const(42), WatInstruction::Drop],
            )
            .unwrap();

        // Check replaced instructions were returned correctly
        assert_eq!(old_instructions.len(), 3);
        assert_eq!(old_instructions[0], WatInstruction::I32Const(2));
        assert_eq!(old_instructions[1], WatInstruction::I32Add);
        assert_eq!(old_instructions[2], WatInstruction::I32Const(3));

        let expected_wat = r#"
            (func $test
                i32.const 1
                i32.const 42
                drop
                i32.mul
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);
    }

    #[test]
    fn test_analyze_stack_state_for_arguments() {
        let mut module = create_test_module();
        let instructions = vec![
            WatInstruction::I32Const(10),
            WatInstruction::I32Const(20),
            WatInstruction::I32Add,
            WatInstruction::Call("$foo".to_string()),
        ];

        module
            .get_function_mut("bar")
            .unwrap()
            .set_body(instructions);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("bar").unwrap();

        // Move to the Call instruction
        while cursor.current_instruction() != Some(WatInstruction::Call("$foo".to_string())) {
            cursor.next();
        }

        let state = cursor.analyze_stack_state_for_arguments().unwrap();
        assert_eq!(state.len(), 1);
        let types: Vec<_> = state.types.iter().collect();
        assert_eq!(types[0], &WasmType::I32);
    }

    #[test]
    fn test_replace_till_the_end_of_block() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                i32.const 1
                i32.const 2
                i32.add
                i32.const 3
                i32.mul
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();
        cursor.next(); // Move to i32.const 1
        cursor.next(); // Move to i32.const 2

        let new_instructions = vec![WatInstruction::I32Const(42), WatInstruction::I32Const(43)];

        let old_instructions = cursor
            .replace_till_the_end_of_the_block(new_instructions)
            .unwrap();

        // Check replaced instructions were returned correctly
        assert_eq!(old_instructions.len(), 4);
        assert_eq!(old_instructions[0], WatInstruction::I32Const(2));
        assert_eq!(old_instructions[1], WatInstruction::I32Add);
        assert_eq!(old_instructions[2], WatInstruction::I32Const(3));
        assert_eq!(old_instructions[3], WatInstruction::I32Mul);

        let expected_wat = r#"
            (func $test
                i32.const 1
                i32.const 42
                i32.const 43
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);
    }

    #[test]
    fn test_replace_till_the_end_of_function_try_catch_aware_simple() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                call $before-try-catch
                try
                    call $before-foo
                    call $foo
                    call $after-foo
                catch $Error
                    call $inside-catch
                end
                call $after-try-catch
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();
        cursor.next(); // call $before-try-catch
        cursor.next(); // try
        let mut iterator = cursor.enter_block().unwrap();
        iterator.next(&mut cursor);
        cursor.next(); // call $before-foo
        cursor.next(); // call $foo

        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::call("$foo"))
        );

        let old_instructions = cursor
            .replace_till_the_end_of_function_try_catch_aware(vec![WatInstruction::call(
                "$new-call",
            )])
            .unwrap();

        let expected_wat = r#"
            (func $test
                call $before-try-catch
                try
                    call $before-foo
                    call $new-call
                catch $Error
                    call $inside-catch
                end
                call $after-try-catch
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);

        let expected_old_instructions_wat = r#"
            (func $old-instructions
                try
                    call $foo
                    call $after-foo
                catch $Error
                    call $inside-catch
                end
                call $after-try-catch
            )"#;

        let expcted_func = parse_wat_function(expected_old_instructions_wat);
        let mut actual_func = WatFunction::new("old-instructions");
        actual_func.set_body(old_instructions);

        assert_functions_eq(&expcted_func, &actual_func);
    }

    #[test]
    fn test_replace_till_the_end_of_function_nested_try_catch() {
        let mut module = create_test_module();

        let input_wat = r#"
            (func $test
                call $outer-start
                try
                    call $outer-try-start
                    try
                        call $inner-try-start
                        call $target-instruction
                        call $inner-try-end
                    catch $InnerError
                        call $inner-catch
                    end
                    call $outer-try-end
                catch $OuterError
                    call $outer-catch
                end
                call $after-try-catch
            )"#;

        let input_func = parse_wat_function(input_wat);
        module.add_function(input_func);

        let mut cursor = InstructionsCursor::new(&mut module);
        cursor.set_current_function("test").unwrap();
        cursor.next(); // call $outer-start
        cursor.next(); // outer try
        let mut outer_iterator = cursor.enter_block().unwrap();
        outer_iterator.next(&mut cursor);
        cursor.next(); // call $outer-try-start
        cursor.next(); // inner try
        let mut inner_iterator = cursor.enter_block().unwrap();
        inner_iterator.next(&mut cursor);
        cursor.next(); // call $inner-try-start
        cursor.next(); // call $target-instruction

        assert_eq!(
            cursor.current_instruction(),
            Some(WatInstruction::call("$target-instruction"))
        );

        let old_instructions = cursor
            .replace_till_the_end_of_function_try_catch_aware(vec![WatInstruction::call(
                "$replacement-call",
            )])
            .unwrap();

        let expected_wat = r#"
            (func $test
                call $outer-start
                try
                    call $outer-try-start
                    try
                        call $inner-try-start
                        call $replacement-call
                    catch $InnerError
                        call $inner-catch
                    end
                    call $outer-try-end
                catch $OuterError
                    call $outer-catch
                end
                call $after-try-catch
            )"#;

        let expected = parse_wat_function(expected_wat);
        let actual = module.get_function("test").unwrap();

        assert_functions_eq(&expected, actual);

        let expected_old_instructions_wat = r#"
            (func $old-instructions
                try
                    try
                        call $target-instruction
                        call $inner-try-end
                    catch $InnerError
                        call $inner-catch
                    end
                    call $outer-try-end
                catch $OuterError
                    call $outer-catch
                end
                call $after-try-catch
            )"#;

        let expected_func = parse_wat_function(expected_old_instructions_wat);
        let mut actual_func = WatFunction::new("old-instructions");
        actual_func.set_body(old_instructions);

        assert_functions_eq(&expected_func, &actual_func);
    }
}
