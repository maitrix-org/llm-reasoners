(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l e g)
(:init 
(harmony)
(planet l)
(planet e)
(planet g)
(province l)
(province e)
(province g)
)
(:goal
(and
(craves l e)
(craves e g)
)))