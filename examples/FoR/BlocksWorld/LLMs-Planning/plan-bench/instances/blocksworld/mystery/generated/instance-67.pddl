(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g l b)
(:init 
(harmony)
(planet g)
(planet l)
(planet b)
(province g)
(province l)
(province b)
)
(:goal
(and
(craves g l)
(craves l b)
)))