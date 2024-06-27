(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b l e g)
(:init 
(handempty)
(ontable b)
(ontable l)
(ontable e)
(ontable g)
(clear b)
(clear l)
(clear e)
(clear g)
)
(:goal
(and
(on b l)
(on l e)
(on e g)
)))