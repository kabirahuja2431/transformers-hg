# English question "have" data


This dataset is very similar to the data used in McCoy et al. (2020) but instead of (quite unnatural) declaratives with _do_-support, all clauses are in present perfect tense.

## Transformations

1. Declaratives

    **Transformation**: Copy declarative sentences.
    
    Sentence types:
    
    (All verbs use _have_/_has_/_haven't_/_hasn't_ as auxiliaries.)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
         _The zebra has annoyed my yak ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
         _The zebra that hasn't chuckled has annoyed my yak ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

         _The zebra has annoyed my yak that hasn't chuckled ._
           
2. Questions

    **Transformation**: Turn declarative sentence into question.
    
    Sentence types:
    
    (All verbs use _have_/_has_/_haven't_/_hasn't_ as auxiliaries.)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
        _The zebra hasn't annoyed my yak ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
        _The zebra that hasn't chuckled has annoyed my yak ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

        _The zebra has annoyed my yak that hasn't chuckled ._

## Files

* `question_have.train(.json)`: 100,000 training examples; includes all 3 types of declarative transformations (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question_have.dev(.json)`: 1,000 development examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question_have.test(.json)`: 10,000 test examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question_have.gen(.json)`: 10,000 generalization examples; includes only RelCl-on-subj examples; _have_ is matched for person in the relative clause and the main clause, either the relative clause or the main clause is negated

