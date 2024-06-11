(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g a e)
(:init 
(handempty)
(ontable g)
(ontable a)
(ontable e)
(clear g)
(clear a)
(clear e)
)
(:goal
(and
(on g a)
(on a e)
)))