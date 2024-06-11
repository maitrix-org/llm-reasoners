(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l e j)
(:init 
(handempty)
(ontable l)
(ontable e)
(ontable j)
(clear l)
(clear e)
(clear j)
)
(:goal
(and
(on l e)
(on e j)
)))