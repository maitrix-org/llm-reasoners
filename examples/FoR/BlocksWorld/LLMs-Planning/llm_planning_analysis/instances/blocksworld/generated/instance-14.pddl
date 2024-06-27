(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l e g)
(:init 
(handempty)
(ontable l)
(ontable e)
(ontable g)
(clear l)
(clear e)
(clear g)
)
(:goal
(and
(on l e)
(on e g)
)))