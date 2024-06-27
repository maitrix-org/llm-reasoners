(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f k l b)
(:init 
(handempty)
(ontable f)
(ontable k)
(ontable l)
(ontable b)
(clear f)
(clear k)
(clear l)
(clear b)
)
(:goal
(and
(on f k)
(on k l)
(on l b)
)))