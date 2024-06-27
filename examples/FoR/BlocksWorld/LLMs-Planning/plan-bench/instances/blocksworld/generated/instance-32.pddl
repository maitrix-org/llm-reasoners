(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f g d)
(:init 
(handempty)
(ontable f)
(ontable g)
(ontable d)
(clear f)
(clear g)
(clear d)
)
(:goal
(and
(on f g)
(on g d)
)))