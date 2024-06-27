(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g d l i)
(:init 
(harmony)
(planet g)
(planet d)
(planet l)
(planet i)
(province g)
(province d)
(province l)
(province i)
)
(:goal
(and
(craves g d)
(craves d l)
(craves l i)
)))