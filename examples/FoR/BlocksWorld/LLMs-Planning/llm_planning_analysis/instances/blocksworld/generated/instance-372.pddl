(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k a b)
(:init 
(handempty)
(ontable k)
(ontable a)
(ontable b)
(clear k)
(clear a)
(clear b)
)
(:goal
(and
(on k a)
(on a b)
)))