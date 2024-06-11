(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l k e g)
(:init 
(handempty)
(ontable l)
(ontable k)
(ontable e)
(ontable g)
(clear l)
(clear k)
(clear e)
(clear g)
)
(:goal
(and
(on l k)
(on k e)
(on e g)
)))