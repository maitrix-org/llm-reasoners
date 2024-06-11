(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e j i)
(:init 
(handempty)
(ontable e)
(ontable j)
(ontable i)
(clear e)
(clear j)
(clear i)
)
(:goal
(and
(on e j)
(on j i)
)))