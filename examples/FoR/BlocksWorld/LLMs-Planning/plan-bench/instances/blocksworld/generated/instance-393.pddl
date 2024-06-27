(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g h a k)
(:init 
(handempty)
(ontable g)
(ontable h)
(ontable a)
(ontable k)
(clear g)
(clear h)
(clear a)
(clear k)
)
(:goal
(and
(on g h)
(on h a)
(on a k)
)))