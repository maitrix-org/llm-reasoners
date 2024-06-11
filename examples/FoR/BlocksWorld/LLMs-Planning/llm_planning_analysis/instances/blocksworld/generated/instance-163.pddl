(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k e i h)
(:init 
(handempty)
(ontable k)
(ontable e)
(ontable i)
(ontable h)
(clear k)
(clear e)
(clear i)
(clear h)
)
(:goal
(and
(on k e)
(on e i)
(on i h)
)))