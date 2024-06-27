(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a g h c)
(:init 
(handempty)
(ontable a)
(ontable g)
(ontable h)
(ontable c)
(clear a)
(clear g)
(clear h)
(clear c)
)
(:goal
(and
(on a g)
(on g h)
(on h c)
)))