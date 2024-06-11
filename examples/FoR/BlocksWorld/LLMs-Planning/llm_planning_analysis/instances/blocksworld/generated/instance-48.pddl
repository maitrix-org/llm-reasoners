(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l e a)
(:init 
(handempty)
(ontable l)
(ontable e)
(ontable a)
(clear l)
(clear e)
(clear a)
)
(:goal
(and
(on l e)
(on e a)
)))