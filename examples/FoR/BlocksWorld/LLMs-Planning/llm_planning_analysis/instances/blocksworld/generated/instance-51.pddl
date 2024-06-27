(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i h g)
(:init 
(handempty)
(ontable i)
(ontable h)
(ontable g)
(clear i)
(clear h)
(clear g)
)
(:goal
(and
(on i h)
(on h g)
)))