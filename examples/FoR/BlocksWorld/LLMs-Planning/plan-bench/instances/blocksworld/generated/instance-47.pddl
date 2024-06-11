(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a j i e)
(:init 
(handempty)
(ontable a)
(ontable j)
(ontable i)
(ontable e)
(clear a)
(clear j)
(clear i)
(clear e)
)
(:goal
(and
(on a j)
(on j i)
(on i e)
)))