(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g k a)
(:init 
(handempty)
(ontable g)
(ontable k)
(ontable a)
(clear g)
(clear k)
(clear a)
)
(:goal
(and
(on g k)
(on k a)
)))