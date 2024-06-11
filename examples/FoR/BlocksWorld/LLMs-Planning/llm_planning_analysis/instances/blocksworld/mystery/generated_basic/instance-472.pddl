

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(planet b)
(craves c e)
(planet d)
(craves e d)
(province a)
(province b)
)
(:goal
(and
(craves a b)
(craves c e)
(craves d a))
)
)


