# English passivization data

An English dataset comparable to the format by Mulligan et al. (2021).

## Transformations

1. Declaratives

    **Transformation**: Copy active sentences.
    
    Sentence types:
            
    * Transitive sentences without prepositional phrase: (No PP)
        
         _ The yak annoyed my walrus ._
        
    * Transitive sentences with PP attached to subject NP (PP-on-subj):
        
         _The yak by our zebra annoyed my walrus ._
        
    * Transitive sentences with PP attached to object NP (PP-on-obj):

         _The yak annoyed my walrus by our zebra ._
           
2. Passivization

    **Transformation**: Turn active sentence into passive sentence.
    
    Sentence types:
            
    * Transitive sentences without prepositional phrase: (No PP)
        
         _ The yak annoyed my walrus ._
        
   * Transitive sentences with PP attached to subject NP (PP-on-subj):
       
        _The yak by our zebra annoyed my walrus ._
       
   * Transitive sentences with PP attached to object NP (PP-on-obj):

        _The yak annoyed my walrus by our zebra ._
   

## Files

* `passive_en_nps.train(.json)`: 100,000 training examples; includes all 3 types of declarative transformations (No PP, PP-on-subj, PP-obj) as well as No-PP and PP-on-obj passivization transformations. 
* `passive_en_nps.dev(.json)`: 10,000 development examples; includes all 3 types of declarative transformations (No PP, PP-on-subj, PP-obj) as well as No-PP and PP-on-obj passivization transformations. 
* `passive_en_nps.test(.json)`: 10,000 test examples; includes all 3 types of declarative transformations (No PP, PP-on-subj, PP-obj) as well as No-PP and PP-on-obj passivization transformations. 
* `passive_en_nps.gen(.json)`: 10,000 generalization examples; includes only PP-on-subj passivization examples.

