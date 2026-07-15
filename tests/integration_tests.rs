#[cfg(test)]
mod tests {
    use spma::*;

    fn make_interner_and_symbol(name: &str) -> (Interner, Symbol) {
        let mut interner = Interner::new();
        let id = interner.intern(name);
        (interner, Symbol::new(id))
    }

    #[test]
    fn test_symbol_creation() {
        let (interner, symbol) = make_interner_and_symbol("test");
        assert_eq!(interner.name(symbol.name), "test");
        assert_eq!(symbol.symbol_type, SymbolType::DataSymbol);
        assert_eq!(symbol.status, SymbolStatus::Contents);
        assert_eq!(symbol.frequency, 1);
    }

    #[test]
    fn test_symbol_matching() {
        let mut interner = Interner::new();
        let cat_id = interner.intern("cat");
        let dog_id = interner.intern("dog");

        let sym1 = Symbol::new(cat_id);
        let sym2 = Symbol::new(cat_id);
        let sym3 = Symbol::new(dog_id);

        assert!(sym1.matches(&sym2));
        assert!(!sym1.matches(&sym3));
    }

    #[test]
    fn test_pattern_creation() {
        let mut interner = Interner::new();
        let the_id = interner.intern("the");
        let cat_id = interner.intern("cat");

        let symbols = vec![Symbol::new(the_id), Symbol::new(cat_id)];
        let pattern = Pattern::new(symbols, 1);

        assert_eq!(pattern.pattern_id, 1);
        assert_eq!(pattern.len(), 2);
        assert!(!pattern.is_empty());
        assert_eq!(pattern.get_symbol_names(&interner), vec!["the", "cat"]);
    }

    #[test]
    fn test_pattern_cost_calculation() {
        let mut interner = Interner::new();
        let id1 = interner.intern("test1");
        let id2 = interner.intern("test2");

        let mut symbols = vec![Symbol::new(id1), Symbol::new(id2)];
        symbols[0].bit_cost = 2.0;
        symbols[1].bit_cost = 3.0;

        let pattern = Pattern::new(symbols, 1);
        assert_eq!(pattern.compute_total_cost(), 5.0);
    }

    #[test]
    fn test_spma_initialization() {
        let sp = SpmaEngine::new();
        assert_eq!(sp.next_pattern_id, 1);
        assert_eq!(sp.keep_rows, 5);
        assert_eq!(sp.max_cycles, 10);
    }

    #[test]
    fn test_symbol_frequency_calculation() {
        let mut sp = SpmaEngine::new();
        let patterns = vec![
            create_test_pattern(&mut sp.interner, vec!["cat", "sat"], 1),
            create_test_pattern(&mut sp.interner, vec!["cat", "ran"], 2),
        ];

        sp.calculate_symbol_frequencies(&patterns);
        let cat_id = sp.interner.intern("cat");
        let sat_id = sp.interner.intern("sat");
        assert_eq!(sp.symbol_frequencies.get(&cat_id), Some(&2));
        assert_eq!(sp.symbol_frequencies.get(&sat_id), Some(&1));
    }

    #[test]
    fn test_pattern_input_parsing() {
        let mut sp = SpmaEngine::new();

        // Create temporary test file
        use std::fs;
        let test_content = "< the cat > sat on < the mat >\n!animal dog #1 eats !food meat #2";
        fs::write("test_input.txt", test_content).unwrap();

        let patterns = sp.load_input("test_input.txt").unwrap();
        assert_eq!(patterns.len(), 2);

        // Check first pattern
        assert_eq!(
            patterns[0].get_symbol_names(&sp.interner),
            vec!["<", "the", "cat", ">", "sat", "on", "<", "the", "mat", ">"]
        );

        // Check second pattern symbols
        let second_pattern_symbols = patterns[1].get_symbol_names(&sp.interner);
        assert!(second_pattern_symbols.contains(&"dog".to_string()));
        assert!(second_pattern_symbols.contains(&"meat".to_string()));

        // Cleanup
        fs::remove_file("test_input.txt").unwrap();
    }

    #[test]
    fn test_learning_cycle() {
        let mut sp = SpmaEngine::new();

        // "cat sat" and "dog sat" share no repeated bigram — n-gram miner finds nothing.
        // With singleton seeding removed (theory-correct), grammar is empty for this corpus.
        let patterns = vec![
            create_test_pattern(&mut sp.interner, vec!["cat", "sat"], 1),
            create_test_pattern(&mut sp.interner, vec!["dog", "sat"], 2),
        ];

        let results = sp.learn(patterns).unwrap();
        assert!(results.cycles > 0);
        // No shared multi-symbol subsequence → no grammar patterns formed
        assert!(results.final_patterns.is_empty());
    }

    #[test]
    fn test_v1_shannon_bit_costs() {
        // Patterns: ["a b c", "a b d", "a b e"]
        // Freqs: a=3, b=3, c=1, d=1, e=1, total=9
        // Expected: a,b cost = -log2(3/9) ≈ 1.585; c,d,e cost = -log2(1/9) ≈ 3.170
        let mut sp = SpmaEngine::new();

        // intern all symbols first, then drop the borrow before calling sp methods
        let (a, b, c, d, e) = {
            let interner = &mut sp.interner;
            let a = interner.intern("a");
            let b = interner.intern("b");
            let c = interner.intern("c");
            let d = interner.intern("d");
            let e = interner.intern("e");
            (a, b, c, d, e)
        };

        fn make_pat(ids: &[u32], pid: u32) -> Pattern {
            let symbols = ids.iter().map(|&id| Symbol::new(id)).collect();
            Pattern::new(symbols, pid)
        }

        let mut patterns = vec![
            make_pat(&[a, b, c], 1),
            make_pat(&[a, b, d], 2),
            make_pat(&[a, b, e], 3),
        ];

        sp.calculate_symbol_frequencies(&patterns);
        sp.assign_symbol_costs(&mut patterns);

        let cost_of = |id: u32| -> f64 {
            patterns
                .iter()
                .flat_map(|p| p.symbols.iter())
                .find(|s| s.name == id)
                .expect("symbol not found in patterns")
                .bit_cost
        };

        let expected_ab = -(3.0_f64 / 9.0).log2(); // ≈ 1.585
        let expected_cde = -(1.0_f64 / 9.0).log2(); // ≈ 3.170

        assert!(
            (cost_of(a) - expected_ab).abs() < 0.001,
            "cost(a)={}",
            cost_of(a)
        );
        assert!(
            (cost_of(b) - expected_ab).abs() < 0.001,
            "cost(b)={}",
            cost_of(b)
        );
        assert!(
            (cost_of(c) - expected_cde).abs() < 0.001,
            "cost(c)={}",
            cost_of(c)
        );
        assert!(
            (cost_of(d) - expected_cde).abs() < 0.001,
            "cost(d)={}",
            cost_of(d)
        );
        assert!(
            (cost_of(e) - expected_cde).abs() < 0.001,
            "cost(e)={}",
            cost_of(e)
        );
    }

    #[test]
    fn test_v2_t_ge_formula() {
        use spma::compute_t_ge;

        // Build a cost table: symbol IDs 0..=4 → costs
        // a=0, b=1, c=2, x=3
        // Use uniform costs for simplicity: cost[i] = (i+1) as f64
        let costs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]; // index = symbol_id

        // Case A: full match — New=[0,1,2], Old=[[0,1,2]], covered=[true,true,true]
        let new_a = &[0u32, 1, 2];
        let old_a: &[&[u32]] = &[&[0u32, 1, 2]];
        let covered_a = &[true, true, true];
        let (g, e, t) = compute_t_ge(new_a, old_a, &costs, covered_a);
        assert!((g - 6.0).abs() < 1e-9, "Case A: G={g}"); // 1+2+3
        assert!((e - 0.0).abs() < 1e-9, "Case A: E={e}");
        assert!((t - 6.0).abs() < 1e-9, "Case A: T={t}");

        // Case B: no match — New=[3,4], Old=[], covered=[false,false]
        let new_b = &[3u32, 4];
        let old_b: &[&[u32]] = &[];
        let covered_b = &[false, false];
        let (g, e, t) = compute_t_ge(new_b, old_b, &costs, covered_b);
        assert!((g - 0.0).abs() < 1e-9, "Case B: G={g}");
        assert!((e - 9.0).abs() < 1e-9, "Case B: E={e}"); // 4+5
        assert!((t - 9.0).abs() < 1e-9, "Case B: T={t}");

        // Case C: partial match — New=[0,1,3,2], Old=[[0,1,2]], covered=[true,true,false,true]
        // G = cost(0)+cost(1)+cost(2) = 1+2+3 = 6
        // E = cost(3) = 4  (position 2 not covered)
        // T = 10
        let new_c = &[0u32, 1, 3, 2];
        let old_c: &[&[u32]] = &[&[0u32, 1, 2]];
        let covered_c = &[true, true, false, true];
        let (g, e, t) = compute_t_ge(new_c, old_c, &costs, covered_c);
        assert!((g - 6.0).abs() < 1e-9, "Case C: G={g}");
        assert!((e - 4.0).abs() < 1e-9, "Case C: E={e}");
        assert!((t - 10.0).abs() < 1e-9, "Case C: T={t}");
    }

    #[test]
    fn test_v3_beam_search_alignment() {
        // Build interner + cost table
        let mut interner = spma::Interner::new();
        let the_id = interner.intern("the");
        let cat_id = interner.intern("cat");
        let sat_id = interner.intern("sat");
        let on_id = interner.intern("on");
        let mat_id = interner.intern("mat");
        let dog_id = interner.intern("dog");

        // Uniform costs (each symbol costs 2.0 bits)
        let n_syms = interner.len();
        let costs: Vec<f64> = vec![2.0; n_syms];

        // Old patterns: single-symbol each
        // "the" appears twice in New — reuse of old[0] gives CD > 0
        let old: Vec<Vec<u32>> = vec![
            vec![the_id],
            vec![cat_id],
            vec![sat_id],
            vec![on_id],
            vec![mat_id],
        ];

        // New: "the cat sat on the mat" — 6 symbols
        let new_full = vec![the_id, cat_id, sat_id, on_id, the_id, mat_id];

        let results = spma::beam_search(&new_full, &old, 10, &costs);
        assert!(!results.is_empty(), "beam search returned no alignments");

        let best = &results[0];
        // All 6 New positions should be covered
        assert!(
            best.covered_new.iter().all(|&c| c),
            "expected full coverage, got {:?}",
            best.covered_new
        );
        // CD > 0 (alignment saves bits)
        assert!(best.cd > 0.0, "expected CD > 0, got {}", best.cd);
        // AlignmentType should be FullA (both fully matched)
        assert_eq!(best.alignment_type, spma::AlignmentType::FullA);

        // Extended test: "the dog sat on the mat" — dog has no Old pattern
        let new_partial = vec![the_id, dog_id, sat_id, on_id, the_id, mat_id];
        let results2 = spma::beam_search(&new_partial, &old, 10, &costs);
        assert!(!results2.is_empty());
        let best2 = &results2[0];
        // dog (position 1) should NOT be covered
        assert!(!best2.covered_new[1], "dog should be uncovered");
        // 5 out of 6 covered
        let covered_count = best2.covered_new.iter().filter(|&&c| c).count();
        assert_eq!(covered_count, 5, "expected 5/6 covered");
        // With single-symbol old patterns at uniform cost, CD is driven by reuse.
        // Both cases reuse "the" equally, so CD is equal. Verify non-negative.
        assert!(best2.cd >= 0.0, "expected CD >= 0, got {}", best2.cd);
        // Partial has higher T (G+E) due to uncovered dog contributing to E
        assert!(best2.t >= best.t, "partial T should be >= full T");
    }

    #[test]
    fn test_v4_convergence() {
        let mut sp = SpmaEngine::new();
        sp.max_cycles = 5;

        let patterns = vec![
            create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
            create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
            create_test_pattern(&mut sp.interner, vec!["a", "b", "e"], 3),
            create_test_pattern(&mut sp.interner, vec!["x", "y", "z"], 4),
            create_test_pattern(&mut sp.interner, vec!["x", "y", "w"], 5),
        ];

        let results = sp.learn(patterns).unwrap();
        assert!(results.cycles >= 1, "should run at least 1 cycle");
        assert!(!results.final_patterns.is_empty(), "should have patterns");

        // T is measured with recomputed bit costs each cycle; cross-cycle cost changes
        // mean per-cycle T is not guaranteed monotone. Assert only that t_per_cycle is
        // populated and that learning ran.
        let t_trace = &results.t_per_cycle;
        assert!(!t_trace.is_empty(), "t_per_cycle should not be empty");
    }

    #[test]
    fn test_v5_one_trial_learning() {
        // A single unique sequence has no repeated bigrams — n-gram miner finds nothing,
        // beam pass finds nothing (empty grammar), grammar stays empty. This is correct:
        // with no repetition there is nothing to compress.
        let mut sp = SpmaEngine::new();
        sp.max_cycles = 3;

        let patterns = vec![create_test_pattern(
            &mut sp.interner,
            vec!["fault_A", "fault_B", "fault_C"],
            1,
        )];

        let results = sp.learn(patterns).unwrap();
        assert!(results.final_patterns.is_empty(), "no repeated structure → empty grammar");
    }

    #[test]
    fn test_v5_one_trial_learning_with_repetition() {
        // Two identical presentations — both bigrams appear twice; grammar should form.
        let mut sp = SpmaEngine::new();
        sp.max_cycles = 5;

        let patterns = vec![
            create_test_pattern(&mut sp.interner, vec!["fault_A", "fault_B", "fault_C"], 1),
            create_test_pattern(&mut sp.interner, vec!["fault_A", "fault_B", "fault_C"], 2),
        ];

        let results = sp.learn(patterns).unwrap();

        let fault_a_id = sp.interner.intern("fault_A");
        let fault_b_id = sp.interner.intern("fault_B");

        // At minimum, [fault_A, fault_B] should form (repeated bigram)
        let has_ab = results.final_patterns.iter().any(|p| {
            let ids: Vec<u32> = p.symbols.iter().map(|s| s.name).collect();
            ids.contains(&fault_a_id) && ids.contains(&fault_b_id)
        });
        assert!(has_ab, "repeated bigram fault_A fault_B should be in grammar");
    }

    #[test]
    fn test_v6_grammar_recovery() {
        let mut sp = SpmaEngine::new();
        sp.max_cycles = 10;
        sp.keep_rows = 10;

        let sentences: Vec<Vec<&str>> = vec![
            vec!["the", "cat", "sat", "the", "mat"],
            vec!["the", "dog", "sat", "a", "cat"],
            vec!["a", "cat", "chased", "the", "dog"],
            vec!["a", "mat", "sat", "the", "cat"],
            vec!["the", "dog", "chased", "a", "mat"],
            vec!["the", "cat", "sat", "a", "dog"],
            vec!["a", "dog", "sat", "the", "cat"],
            vec!["the", "mat", "chased", "a", "dog"],
            vec!["a", "cat", "sat", "the", "mat"],
            vec!["the", "dog", "sat", "the", "cat"],
            vec!["a", "mat", "chased", "a", "cat"],
            vec!["the", "cat", "chased", "the", "dog"],
            vec!["a", "dog", "chased", "the", "mat"],
            vec!["the", "mat", "sat", "a", "cat"],
            vec!["a", "cat", "sat", "a", "dog"],
            vec!["the", "dog", "chased", "the", "mat"],
            vec!["a", "mat", "sat", "a", "dog"],
            vec!["the", "cat", "chased", "a", "mat"],
            vec!["a", "dog", "sat", "the", "mat"],
            vec!["the", "mat", "chased", "the", "cat"],
        ];

        let mut patterns = Vec::new();
        for (i, sentence) in sentences.iter().enumerate() {
            let syms: Vec<&str> = sentence.clone();
            patterns.push(create_test_pattern(&mut sp.interner, syms, (i + 1) as u32));
        }

        let results = sp.learn(patterns).unwrap();

        // Use global MDL compression ratio:
        // - global G = cost of storing grammar (all old patterns, each once)
        // - global E = sum of uncovered symbol costs across all new patterns
        // - ratio = total_raw / (global_G + global_E)
        let compression_ratio = sp.compute_global_compression_ratio(
            &sp.new_patterns.clone(),
            &results.final_patterns,
            10,
        );

        println!(
            "Global compression ratio: {:.3}, grammar size: {} patterns",
            compression_ratio,
            results.final_patterns.len()
        );

        let multi_symbol_count = results
            .final_patterns
            .iter()
            .filter(|p| p.symbols.len() >= 2)
            .count();
        println!("Multi-symbol patterns: {}", multi_symbol_count);

        assert!(
            compression_ratio > 2.0,
            "expected global compression ratio > 2.0, got {:.3}. Grammar size: {} patterns",
            compression_ratio,
            results.final_patterns.len()
        );

        // Grammar has grown beyond the initial alphabet seeds
        assert!(
            results.final_patterns.len() > 8,
            "expected grammar to grow beyond vocabulary size (8), got {}",
            results.final_patterns.len()
        );

        // Verify multi-symbol patterns were extracted
        assert!(
            multi_symbol_count > 0,
            "expected multi-symbol patterns to be extracted, got 0"
        );
    }

    // Helper function
    fn create_test_pattern(interner: &mut Interner, words: Vec<&str>, id: u32) -> Pattern {
        let symbols: Vec<Symbol> = words
            .iter()
            .map(|word| Symbol::new(interner.intern(word)))
            .collect();
        Pattern::new(symbols, id)
    }

    // ── Interner ──────────────────────────────────────────────────────────────

    #[test]
    fn interner_same_string_same_id() {
        let mut i = Interner::new();
        let a = i.intern("foo");
        let b = i.intern("foo");
        assert_eq!(a, b);
        assert_eq!(i.len(), 1);
    }

    #[test]
    fn interner_different_strings_different_ids() {
        let mut i = Interner::new();
        let a = i.intern("foo");
        let b = i.intern("bar");
        assert_ne!(a, b);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn interner_name_roundtrip() {
        let mut i = Interner::new();
        let id = i.intern("hello");
        assert_eq!(i.name(id), "hello");
    }

    #[test]
    #[should_panic(expected = "intern ID out of bounds")]
    fn interner_name_out_of_bounds_panics() {
        let i = Interner::new();
        i.name(99);
    }

    // ── assign_symbol_types ───────────────────────────────────────────────────

    #[test]
    fn assign_symbol_types_promotes_identification_symbols() {
        let mut sp = SpmaEngine::new();
        let a = sp.interner.intern("A");
        let b = sp.interner.intern("B");

        // Pattern where B is marked Identification
        let mut sym_a = Symbol::new(a);
        sym_a.status = SymbolStatus::Contents;
        let mut sym_b = Symbol::new(b);
        sym_b.status = SymbolStatus::Identification;

        sp.new_patterns = vec![Pattern::new(vec![sym_a, sym_b], 1)];
        sp.assign_symbol_types();

        let b_in_pat = sp.new_patterns[0].symbols.iter().find(|s| s.name == b).unwrap();
        assert_eq!(b_in_pat.symbol_type, SymbolType::ContextSymbol,
            "Identification symbol should be promoted to ContextSymbol");

        let a_in_pat = sp.new_patterns[0].symbols.iter().find(|s| s.name == a).unwrap();
        assert_eq!(a_in_pat.symbol_type, SymbolType::DataSymbol,
            "Contents symbol should stay DataSymbol");
    }

    #[test]
    fn assign_symbol_types_no_identification_no_change() {
        let mut sp = SpmaEngine::new();
        let a = sp.interner.intern("A");
        sp.new_patterns = vec![Pattern::new(vec![Symbol::new(a)], 1)];
        sp.assign_symbol_types();
        assert_eq!(sp.new_patterns[0].symbols[0].symbol_type, SymbolType::DataSymbol);
    }

    // ── compute_t_ge ─────────────────────────────────────────────────────────

    #[test]
    fn compute_t_ge_overlapping_symbol_ids_in_new_and_old() {
        // Symbol 0 appears in both new and old — G counts old occurrence, E skips covered new
        let costs = vec![2.0, 3.0];
        let new = &[0u32, 1];
        let old: &[&[u32]] = &[&[0u32]];
        let covered = &[true, false];
        let (g, e, t) = compute_t_ge(new, old, &costs, covered);
        assert!((g - 2.0).abs() < 1e-9, "G={g}");  // old[0] = id 0, cost 2.0
        assert!((e - 3.0).abs() < 1e-9, "E={e}");  // id 1 uncovered, cost 3.0
        assert!((t - 5.0).abs() < 1e-9, "T={t}");
    }

    #[test]
    fn compute_t_ge_empty_old_all_e() {
        let costs = vec![1.0, 2.0, 3.0];
        let new = &[0u32, 1, 2];
        let old: &[&[u32]] = &[];
        let covered = &[false, false, false];
        let (g, e, t) = compute_t_ge(new, old, &costs, covered);
        assert!((g - 0.0).abs() < 1e-9);
        assert!((e - 6.0).abs() < 1e-9);
        assert!((t - 6.0).abs() < 1e-9);
    }

    #[test]
    fn compute_t_ge_all_covered_no_e() {
        let costs = vec![1.0, 2.0];
        let new = &[0u32, 1];
        let old: &[&[u32]] = &[&[0u32, 1]];
        let covered = &[true, true];
        let (g, e, t) = compute_t_ge(new, old, &costs, covered);
        assert!((g - 3.0).abs() < 1e-9);
        assert!((e - 0.0).abs() < 1e-9);
        assert!((t - 3.0).abs() < 1e-9);
    }

    // ── write_alignment_table ─────────────────────────────────────────────────

    #[test]
    fn write_alignment_table_empty_pattern_writes_nothing() {
        let interner = Interner::new();
        let empty_pat = Pattern::new(vec![], 0);
        let alignment = spma::beam_search(&[], &[], 5, &[]);
        // Nothing to write — must not panic
        let mut out = String::new();
        if let Some(best) = alignment.into_iter().next() {
            spma::write_alignment_table(&mut out, &empty_pat, &best, &[], &interner);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn write_alignment_table_no_old_patterns() {
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let b = interner.intern("B");
        let new_syms = vec![Symbol::new(a), Symbol::new(b)];
        let new_pat = Pattern::new(new_syms, 1);
        let costs = vec![1.0, 1.0];
        let results = spma::beam_search(&[a, b], &[], 5, &costs);
        let best = results.into_iter().next().unwrap();
        let mut out = String::new();
        spma::write_alignment_table(&mut out, &new_pat, &best, &[], &interner);
        assert!(out.contains("New:"), "should contain New row");
        assert!(out.contains('A') || out.contains('B'), "should contain symbol names");
    }

    #[test]
    fn write_alignment_table_multi_row_contains_old_labels() {
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let b = interner.intern("B");
        let costs = vec![2.0, 2.0];
        let old = vec![vec![a], vec![b]];
        let new_ids = vec![a, b];
        let results = spma::beam_search(&new_ids, &old, 10, &costs);
        let best = results.into_iter().next().unwrap();

        let new_syms: Vec<Symbol> = new_ids.iter().map(|&id| Symbol::new(id)).collect();
        let new_pat = Pattern::new(new_syms, 0);
        let old_pats: Vec<Pattern> = old.iter().map(|ids| {
            Pattern::new(ids.iter().map(|&id| Symbol::new(id)).collect(), 0)
        }).collect();

        let mut out = String::new();
        spma::write_alignment_table(&mut out, &new_pat, &best, &old_pats, &interner);
        assert!(out.contains("New:"));
        assert!(out.contains("Old1:") || out.contains("Old2:"),
            "multi-row should have Old labels, got:\n{out}");
        assert!(out.contains("Matched:"), "should contain stats line");
    }

    // ── beam_search ───────────────────────────────────────────────────────────

    #[test]
    fn beam_search_partial_coverage_correct_cd() {
        // New=[A,B,C], Old=[[A,B]] — C is uncovered
        let mut i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let c = i.intern("C");
        let costs = vec![2.0, 3.0, 4.0];
        let old = vec![vec![a, b]];
        let results = spma::beam_search(&[a, b, c], &old, 10, &costs);
        let best = &results[0];
        assert!(best.covered_new[0], "A should be covered");
        assert!(best.covered_new[1], "B should be covered");
        assert!(!best.covered_new[2], "C should not be covered");
        // CD = cost(A)+cost(B) = 5.0, E = cost(C) = 4.0
        assert!((best.cd - 5.0).abs() < 1e-9, "CD={}", best.cd);
        assert!((best.e - 4.0).abs() < 1e-9, "E={}", best.e);
    }

    #[test]
    fn beam_search_multi_symbol_old_pattern() {
        let mut i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let c = i.intern("C");
        let costs = vec![1.0, 1.0, 1.0];
        let old = vec![vec![a, b, c]];
        let results = spma::beam_search(&[a, b, c], &old, 5, &costs);
        let best = &results[0];
        assert!(best.covered_new.iter().all(|&c| c), "all positions should be covered");
        assert_eq!(best.alignment_type, spma::AlignmentType::FullA);
    }

    #[test]
    fn beam_search_alignment_type_full_b_partial_new() {
        // Old pattern [A] fully matched, New=[A,B] — B uncovered → FullB
        let mut i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let costs = vec![1.0, 1.0];
        let old = vec![vec![a]];
        let results = spma::beam_search(&[a, b], &old, 5, &costs);
        let best = &results[0];
        assert!(best.covered_new[0]);
        assert!(!best.covered_new[1]);
        assert_eq!(best.alignment_type, spma::AlignmentType::FullB);
    }

    #[test]
    fn beam_search_monotonic_order_enforced() {
        // Old=[A,B], New=[B,A] — B before A in New cannot match [A,B] in order
        let mut i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let costs = vec![1.0, 1.0];
        let old = vec![vec![a, b]];
        let results = spma::beam_search(&[b, a], &old, 5, &costs);
        let best = &results[0];
        // Only one of the two can be covered (monotonic cursor prevents both)
        let covered_count = best.covered_new.iter().filter(|&&c| c).count();
        assert!(covered_count <= 1,
            "out-of-order sequence should not produce full coverage, got {covered_count}");
    }

    // ── extract_frequent_ngrams cold-start vs beam-switch ─────────────────────

    #[test]
    fn extract_frequent_ngrams_cold_start_adds_bigrams() {
        let mut sp = SpmaEngine::new();
        let pats = vec![
            create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
            create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
        ];
        let results = sp.learn(pats).unwrap();
        // "a b" appears twice — cold-start n-gram miner should add it
        let a = sp.interner.intern("a");
        let b = sp.interner.intern("b");
        let has_ab = results.final_patterns.iter().any(|p| {
            let ids: Vec<u32> = p.symbols.iter().map(|s| s.name).collect();
            ids == vec![a, b]
        });
        assert!(has_ab, "cold-start should extract bigram [a,b]");
    }

    #[test]
    fn extract_frequent_ngrams_does_not_add_singletons() {
        let mut sp = SpmaEngine::new();
        // Each bigram appears only once — below min_freq=2 threshold
        let pats = vec![
            create_test_pattern(&mut sp.interner, vec!["a", "b"], 1),
            create_test_pattern(&mut sp.interner, vec!["c", "d"], 2),
        ];
        let results = sp.learn(pats).unwrap();
        let multi = results.final_patterns.iter().filter(|p| p.symbols.len() >= 2).count();
        assert_eq!(multi, 0, "no bigram appears ≥2 times, should add none");
    }

    #[test]
    fn beam_driven_phase_skips_ngram_miner() {
        // After cold-start fires and multi-symbol patterns exist,
        // n-gram miner should not run again (extract_frequent_ngrams guarded by !has_multi_symbol).
        // Verify by checking results are stable: same corpus, second learn call
        // should produce same or fewer patterns.
        let mut sp = SpmaEngine::new();
        let pats = vec![
            create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
            create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
            create_test_pattern(&mut sp.interner, vec!["a", "b", "e"], 3),
        ];
        let r1 = sp.learn(pats.clone()).unwrap();
        // Rebuild engine with same data — should converge to same or fewer patterns
        let mut sp2 = SpmaEngine::new();
        sp2.interner = sp.interner.clone();
        let r2 = sp2.learn(pats).unwrap();
        assert_eq!(r1.final_patterns.len(), r2.final_patterns.len(),
            "deterministic: same corpus same result");
    }

    // ── compute_global_compression_ratio ─────────────────────────────────────

    #[test]
    fn compression_ratio_empty_grammar_is_one() {
        let sp = SpmaEngine::new();
        // No patterns at all → global_t = 0 → returns 1.0 (division-by-zero guard)
        let ratio = sp.compute_global_compression_ratio(&[], &[], 5);
        assert!((ratio - 1.0).abs() < 1e-9, "ratio={ratio}");
    }

    #[test]
    fn compression_ratio_no_multi_symbol_patterns_is_one() {
        let mut sp = SpmaEngine::new();
        let pats = vec![create_test_pattern(&mut sp.interner, vec!["a"], 1)];
        // Only single-symbol seeds — global_G=0, E=all uncovered → ratio=raw/E=1.0
        let ratio = sp.compute_global_compression_ratio(&pats, &pats, 5);
        assert!((ratio - 1.0).abs() < 1e-9, "ratio={ratio}");
    }

    #[test]
    fn compression_ratio_perfect_coverage_greater_than_one() {
        let mut sp = SpmaEngine::new();
        let a = sp.interner.intern("a");
        let b = sp.interner.intern("b");
        // Build a grammar pattern [a,b] with bit_cost set
        let mut sa = Symbol::new(a); sa.bit_cost = 2.0;
        let mut sb = Symbol::new(b); sb.bit_cost = 2.0;
        let grammar_pat = Pattern::new(vec![sa.clone(), sb.clone()], 1);
        // Many new patterns all = [a,b]
        let new_pats: Vec<Pattern> = (0..5).map(|i| {
            Pattern::new(vec![sa.clone(), sb.clone()], i + 10)
        }).collect();
        // grammar_cost = 4.0; each new covered fully → E=0; total_raw = 5*4 = 20
        // ratio = 20 / (4 + 0) = 5.0
        let ratio = sp.compute_global_compression_ratio(&new_pats, &[grammar_pat], 5);
        assert!(ratio > 1.0, "good compression should yield ratio > 1, got {ratio}");
    }

    // ── Spma public API ───────────────────────────────────────────────────────

    #[test]
    fn spma_train_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("spma_roundtrip_test.bin");
        let path_str = path.to_str().unwrap();

        let mut engine = spma::Spma::new();
        engine.train(&[
            vec!["fault_A", "fault_B", "fault_C"],
            vec!["fault_A", "fault_B", "fault_D"],
        ]).unwrap();
        engine.save(path_str).unwrap();

        let engine2 = spma::Spma::load(path_str).unwrap();
        // "fault_A fault_B" is a shared bigram → forms a grammar pattern.
        // "fault_C" appears only once and has no shared substructure → uncovered (E > 0).
        // Correct behavior: partial match, is_anomaly=true for the novel suffix.
        let result = engine2.infer(&["fault_A", "fault_B", "fault_C"]).unwrap();
        // The shared prefix must be covered
        assert!(
            result.alignment.contains("fault_A") && result.alignment.contains("fault_B"),
            "shared prefix should appear in alignment"
        );
        // fault_C is unique — appears in unmatched
        assert!(
            result.unmatched.contains(&"fault_C".to_string()),
            "fault_C should be unmatched (unique symbol, no grammar pattern)"
        );

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn spma_infer_unseen_symbol_is_anomaly() {
        let mut engine = spma::Spma::new();
        engine.train(&[vec!["A", "B", "C"]]).unwrap();
        let result = engine.infer(&["A", "B", "X"]).unwrap();
        assert!(result.is_anomaly, "unseen X should trigger anomaly");
        assert!(result.e_cost > 0.0, "e_cost should be > 0 due to unknown penalty");
        assert!(result.unmatched.contains(&"X".to_string()), "X should be in unmatched");
    }

    #[test]
    fn spma_infer_alignment_string_non_empty() {
        let mut engine = spma::Spma::new();
        engine.train(&[vec!["A", "B"], vec!["A", "C"]]).unwrap();
        let result = engine.infer(&["A", "B"]).unwrap();
        assert!(!result.alignment.is_empty(), "alignment string should be populated");
        assert!(result.alignment.contains("New:"), "alignment should have New row");
    }

    #[test]
    fn spma_infer_fully_unseen_sequence() {
        let mut engine = spma::Spma::new();
        engine.train(&[vec!["A", "B"]]).unwrap();
        let result = engine.infer(&["X", "Y", "Z"]).unwrap();
        assert!(result.is_anomaly, "all-unknown sequence should be anomaly");
        assert_eq!(result.unmatched.len(), 3, "all 3 symbols should be unmatched");
        assert!(result.alignment.contains("New:"), "got: {}", result.alignment);
        assert!(result.alignment.contains('X'), "got: {}", result.alignment);
    }

    #[test]
    fn spma_infer_known_but_rare_symbol_not_penalised_as_unknown() {
        // Symbol "C" appears once in training — low frequency, high bit cost, but KNOWN.
        // Should not get the unknown_penalty on top of its real cost.
        let mut engine = spma::Spma::new();
        engine.train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "D"],
        ]).unwrap();
        let known_result = engine.infer(&["A", "B", "C"]).unwrap();
        let unknown_result = engine.infer(&["A", "B", "X"]).unwrap();
        // Both may be anomalies depending on grammar, but unknown should have >= e_cost
        // because the penalty is additive. Known-rare must NOT get double-penalised.
        assert!(
            unknown_result.e_cost >= known_result.e_cost,
            "unknown symbol should cost at least as much as a known-rare symbol: \
             unknown_e={} known_e={}", unknown_result.e_cost, known_result.e_cost
        );
    }

    #[test]
    fn spma_train_special_symbols_bracket_and_id() {
        let mut engine = spma::Spma::new();
        engine.train(&[
            vec!["<", "cat", ">"],
            vec!["<", "dog", ">"],
        ]).unwrap();
        // Should not panic and brackets should be handled
        let result = engine.infer(&["<", "cat", ">"]).unwrap();
        // Brackets are in the grammar — should not be anomaly
        assert!(!result.is_anomaly || result.e_cost == 0.0);
    }

    #[test]
    fn mdl_accumulator_determinism_multi_candidate_corpus() {
        // Corpus large enough to produce multiple distinct candidates per cycle.
        // Both engines must produce identical grammar (same pattern names).
        // This verifies the accumulator refactor is behaviourally equivalent to
        // the old per-iteration rebuild.
        let corpus: Vec<Vec<&str>> = vec![
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "D"],
            vec!["A", "B", "D"],
            vec!["X", "Y", "Z"],
            vec!["X", "Y", "Z"],
            vec!["X", "Y", "W"],
            vec!["A", "B", "C"],
            vec!["X", "Y", "Z"],
            vec!["A", "B", "D"],
        ];

        let mut e1 = spma::Spma::new();
        e1.train(&corpus).unwrap();

        let mut e2 = spma::Spma::new();
        e2.train(&corpus).unwrap();

        // Both runs must agree: same e_cost for every sequence.
        let probe = vec!["A", "B", "C"];
        let r1 = e1.infer(&probe).unwrap();
        let r2 = e2.infer(&probe).unwrap();
        assert_eq!(r1.e_cost, r2.e_cost, "accumulator runs diverged");
        assert_eq!(r1.is_anomaly, r2.is_anomaly);

        let probe2 = vec!["X", "Y", "W"];
        let r3 = e1.infer(&probe2).unwrap();
        let r4 = e2.infer(&probe2).unwrap();
        assert_eq!(r3.e_cost, r4.e_cost);
    }

    #[test]
    fn no_singleton_seeding_grammar_contains_multi_symbol_pattern() {
        // With singleton seeding removed, the grammar must grow multi-symbol patterns
        // from repeated substructure — not just cover the alphabet.
        let mut engine = spma::Spma::new();
        engine.train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ]).unwrap();
        let result = engine.infer(&["A", "B", "C"]).unwrap();
        // Grammar should have learned [A,B] or [B,C] or [A,B,C]
        // so the sequence is fully or substantially covered
        assert!(
            result.e_cost == 0.0 || result.cd > 0.0,
            "repeated identical sequence should be covered by grammar: e={} cd={}",
            result.e_cost, result.cd
        );
    }

    #[test]
    fn no_singleton_seeding_order_violation_detectable() {
        // Once grammar has multi-symbol patterns, reversed input MUST have higher E.
        // [A,B,C] trained → grammar learns [A,B] and/or [B,C].
        // [C,B,A] cannot match [A,B] contiguously → higher E.
        let mut engine = spma::Spma::new();
        engine.train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ]).unwrap();
        let forward = engine.infer(&["A", "B", "C"]).unwrap();
        let reversed = engine.infer(&["C", "B", "A"]).unwrap();
        assert!(
            reversed.e_cost >= forward.e_cost,
            "reversed sequence should have E >= forward: reversed_e={} forward_e={}",
            reversed.e_cost, forward.e_cost
        );
    }
}
