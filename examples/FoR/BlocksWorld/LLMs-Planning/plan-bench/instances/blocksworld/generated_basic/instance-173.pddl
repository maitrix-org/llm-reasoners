

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a b)
(on b d)
(ontable c)
(on d c)
(clear a)
)
(:goal
(and
(on a d)
(on b a))
)
)


