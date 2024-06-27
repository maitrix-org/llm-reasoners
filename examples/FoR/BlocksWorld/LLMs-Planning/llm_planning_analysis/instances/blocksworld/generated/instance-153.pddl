(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a h e)
(:init 
(handempty)
(ontable a)
(ontable h)
(ontable e)
(clear a)
(clear h)
(clear e)
)
(:goal
(and
(on a h)
(on h e)
)))