(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e l a)
(:init 
(handempty)
(ontable e)
(ontable l)
(ontable a)
(clear e)
(clear l)
(clear a)
)
(:goal
(and
(on e l)
(on l a)
)))