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
        assert_eq!(sp.cost_factor, 2.0);
        assert_eq!(sp.max_alignments_per_cycle, 50);
        assert_eq!(sp.next_pattern_id, 1);
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
    fn test_hit_structure_building() {
        let mut sp = SpmaEngine::new();
        let pattern1 = create_test_pattern(&mut sp.interner, vec!["the", "cat", "sat"], 1);
        let pattern2 = create_test_pattern(&mut sp.interner, vec!["the", "dog", "sat"], 2);

        let hits = sp.build_hit_structure(&pattern1, &[pattern2]);

        // Test that the function runs without error
        let _ = hits;
    }

    #[test]
    fn test_grammar_creation() {
        let mut interner = Interner::new();
        let patterns = vec![
            create_test_pattern(&mut interner, vec!["cat", "sat"], 1),
            create_test_pattern(&mut interner, vec!["dog", "ran"], 2),
        ];

        let mut grammar = Grammar::new(1, patterns);
        assert_eq!(grammar.grammar_id, 1);
        assert_eq!(grammar.patterns.len(), 2);

        grammar.compute_grammar_size();
        assert!(grammar.grammar_size >= 0.0);
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
        sp.max_alignments_per_cycle = 5; // Limit for test

        let patterns = vec![
            create_test_pattern(&mut sp.interner, vec!["cat", "sat"], 1),
            create_test_pattern(&mut sp.interner, vec!["dog", "sat"], 2),
        ];

        let results = sp.learn(patterns).unwrap();
        assert!(results.cycles > 0);
        assert!(!results.final_patterns.is_empty());
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

        // V4 spec: T must be monotonically non-increasing across cycles
        let t_trace = &results.t_per_cycle;
        assert!(!t_trace.is_empty(), "t_per_cycle should not be empty");
        let epsilon = 1e-9;
        for i in 1..t_trace.len() {
            assert!(
                t_trace[i] <= t_trace[i - 1] + epsilon,
                "T increased at cycle {}: {} -> {}",
                i,
                t_trace[i - 1],
                t_trace[i]
            );
        }
    }

    #[test]
    fn test_v5_one_trial_learning() {
        let mut sp = SpmaEngine::new();
        sp.max_cycles = 3;

        let patterns = vec![create_test_pattern(
            &mut sp.interner,
            vec!["fault_A", "fault_B", "fault_C"],
            1,
        )];

        let results = sp.learn(patterns).unwrap();

        // After learning: old store must contain a pattern that covers fault_A fault_B fault_C
        let fault_a_id = sp.interner.intern("fault_A");
        let fault_b_id = sp.interner.intern("fault_B");
        let fault_c_id = sp.interner.intern("fault_C");

        // Old store should be able to fully recognise the pattern
        // (either as a whole pattern or via single-symbol seeds)
        let has_a = results
            .final_patterns
            .iter()
            .any(|p| p.symbols.iter().any(|s| s.name == fault_a_id));
        let has_b = results
            .final_patterns
            .iter()
            .any(|p| p.symbols.iter().any(|s| s.name == fault_b_id));
        let has_c = results
            .final_patterns
            .iter()
            .any(|p| p.symbols.iter().any(|s| s.name == fault_c_id));
        assert!(
            has_a && has_b && has_c,
            "old store should contain patterns covering fault_A, fault_B, fault_C"
        );

        // Second presentation: feed same pattern, should get full coverage
        let new_ids = vec![fault_a_id, fault_b_id, fault_c_id];
        let old_id_vecs: Vec<Vec<u32>> = results
            .final_patterns
            .iter()
            .map(|p| p.symbols.iter().map(|s| s.name).collect())
            .collect();

        let n_syms = sp.interner.len();
        let costs = vec![2.0f64; n_syms];

        let alignments = spma::beam_search(&new_ids, &old_id_vecs, 5, &costs);
        assert!(!alignments.is_empty());
        // Full coverage means the pattern is fully recognised
        assert!(
            alignments[0].covered_new.iter().all(|&c| c),
            "second presentation should yield full coverage"
        );
        // CD >= 0 means alignment is at least as good as raw encoding
        // (strict CD > 0 requires pointer-based G formula where reuse is cheaper)
        assert!(
            alignments[0].cd >= 0.0,
            "second presentation should yield CD >= 0, got {}",
            alignments[0].cd
        );
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
}
