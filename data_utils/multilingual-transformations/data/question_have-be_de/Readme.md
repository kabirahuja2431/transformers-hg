# German question "have-be" data

A German dataset similar to the English "have" data. All verbs
are in perfect tense and with either _have_ (_hat_) or _be_ (_sein_)
as auxiliaries.

## Transformations

1. Declaratives

    **Transformation**: Copy declarative sentences.
    
    Sentence types:
    
    (All verbs use _have_/_has_/are/is as auxiliaries.)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
         _ Der Wellensittich hat meinen Hund genervt ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
         _Der Wellensittich, der geschwommen ist, hat meinen Hund genervt ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

         _Der Wellensittich hat meinen Hund, der geschwommen ist, genervt ._
           
2. Questions

    **Transformation**: Turn declarative sentence into question.
    
    Sentence types:
    
    (All verbs use _have_/_has_/_haven't_/_hasn't_ as auxiliaries.)
        
    * Transitive and intransitive sentences without relative clauses: (No RelCl)
        
        _ Der Wellensittich hat meinen Hund genervt ._
        
    * Transitive and intransitive sentences with relative clause on subject NP (RelCl-on-subj):
        
         _Der Wellensittich, der geschwommen ist, hat meinen Hund genervt ._
        
    * Transitive and intransitive sentences with relative clause on object NP (RelCl-on-obj):

         _Der Wellensittich hat meinen Hund, der geschwommen ist, genervt ._

## Files

* `question_have.de.train(.json)`: 100,000 training examples; includes all 3 types of declarative transformations (No RelCl, RelCl-on-subj, RelCl-obj) 
* `question_have.de.dev(.json)`: 10,000 development examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) 
* `question_have.de.test(.json)`: 10,000 test examples; includes all 3 types of declarative sentences (No RelCl, RelCl-on-subj, RelCl-obj) 
* `question_have.de.gen_rc_o(.json)`: 10,000 generalization examples; includes only RelCl-on-obj examples; _have_/_be_ are matched for person in the relative clause and the main clause,  the relative clause includes a verb that requires _be_
* `question_have.de.gen_rc_s(.json)`: 10,000 generalization examples; includes only RelCl-on-subj examples; _have_/_be_ are matched for person in the relative clause and the main clause, either the relative clause or the main clause includes a verb that requires _be_
* `question_have.en-de.train(.json)`: 200,000 training examples; combines the 100,00 English training examples from the question_have-havent dataset and the 100,000 German training examples from this dataset.  Half of the English examples are declaratives and half are questions (No RelCl or RelCl-on-obj); the German examples are all declarative sentenes.


