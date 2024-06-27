(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a h g)
(:init 
(handempty)
(ontable a)
(ontable h)
(ontable g)
(clear a)
(clear h)
(clear g)
)
(:goal
(and
(on a h)
(on h g)
)))