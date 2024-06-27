(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d l a)
(:init 
(harmony)
(planet d)
(planet l)
(planet a)
(province d)
(province l)
(province a)
)
(:goal
(and
(craves d l)
(craves l a)
)))