# Dataset from McCoy et al. (2020)


## Transformations

1. Declaratives

    **Transformation**: Copy declarative sentences.
    
    Sentence types:
    
    (All sentences have obligatory do-support)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
         _The zebra does annoy my yak ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
         _The zebra that doesn't chuckle does annoy my yak ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

         _The zebra does annoy my yak that doesn't chuckle ._
           
2. Questions

    **Transformation**: Turn declarative sentence into question.
    
    Sentence types:
    
    (All sentences have obligatory do-support)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
        _The zebra does annoy my yak ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
        _The zebra that doesn't chuckle does annoy my yak ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

        _The zebra does annoy my yak that doesn't chuckle ._

## Files

* `question.train(.json)`: 100,000 training examples; includes all 3 types of declarative transformations (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question.dev(.json)`: 1,000 development examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question.test(.json)`: 10,000 test examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) and the No RelCl and ReclCl-on-obj question transformations.
* `question.gen(.json)`: 10,000 generalization examples; includes only RelCl-on-subj examples


