(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f j g d)
(:init 
(handempty)
(ontable f)
(ontable j)
(ontable g)
(ontable d)
(clear f)
(clear j)
(clear g)
(clear d)
)
(:goal
(and
(on f j)
(on j g)
(on g d)
)))