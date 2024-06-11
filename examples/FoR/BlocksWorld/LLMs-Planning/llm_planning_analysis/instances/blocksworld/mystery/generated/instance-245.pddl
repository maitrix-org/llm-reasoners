(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g l b j)
(:init 
(harmony)
(planet g)
(planet l)
(planet b)
(planet j)
(province g)
(province l)
(province b)
(province j)
)
(:goal
(and
(craves g l)
(craves l b)
(craves b j)
)))