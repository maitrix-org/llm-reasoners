(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b d e g)
(:init 
(handempty)
(ontable b)
(ontable d)
(ontable e)
(ontable g)
(clear b)
(clear d)
(clear e)
(clear g)
)
(:goal
(and
(on b d)
(on d e)
(on e g)
)))