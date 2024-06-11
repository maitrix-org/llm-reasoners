(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g d c)
(:init 
(handempty)
(ontable g)
(ontable d)
(ontable c)
(clear g)
(clear d)
(clear c)
)
(:goal
(and
(on g d)
(on d c)
)))