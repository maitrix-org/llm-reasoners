(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a g h k)
(:init 
(handempty)
(ontable a)
(ontable g)
(ontable h)
(ontable k)
(clear a)
(clear g)
(clear h)
(clear k)
)
(:goal
(and
(on a g)
(on g h)
(on h k)
)))