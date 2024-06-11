

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(planet c)
(planet d)
(province a)
(province b)
(province d)
)
(:goal
(and
(craves a b)
(craves b d)
(craves d c))
)
)


