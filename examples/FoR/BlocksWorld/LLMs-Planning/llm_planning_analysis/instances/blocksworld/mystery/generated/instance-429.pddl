(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l g b)
(:init 
(harmony)
(planet l)
(planet g)
(planet b)
(province l)
(province g)
(province b)
)
(:goal
(and
(craves l g)
(craves g b)
)))