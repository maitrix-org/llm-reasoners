(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i k h g)
(:init 
(handempty)
(ontable i)
(ontable k)
(ontable h)
(ontable g)
(clear i)
(clear k)
(clear h)
(clear g)
)
(:goal
(and
(on i k)
(on k h)
(on h g)
)))