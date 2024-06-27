

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(craves b c)
(planet c)
(planet d)
(province a)
(province d)
)
(:goal
(and
(craves a c)
(craves b d)
(craves d a))
)
)


