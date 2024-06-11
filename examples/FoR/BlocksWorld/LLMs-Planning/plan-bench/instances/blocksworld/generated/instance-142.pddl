(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e g c h)
(:init 
(handempty)
(ontable e)
(ontable g)
(ontable c)
(ontable h)
(clear e)
(clear g)
(clear c)
(clear h)
)
(:goal
(and
(on e g)
(on g c)
(on c h)
)))