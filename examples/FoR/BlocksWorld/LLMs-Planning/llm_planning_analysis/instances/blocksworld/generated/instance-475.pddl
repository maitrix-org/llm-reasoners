(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g k h e)
(:init 
(handempty)
(ontable g)
(ontable k)
(ontable h)
(ontable e)
(clear g)
(clear k)
(clear h)
(clear e)
)
(:goal
(and
(on g k)
(on k h)
(on h e)
)))