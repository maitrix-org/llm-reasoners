

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b a)
(ontable c)
(on d c)
(clear b)
(clear d)
)
(:goal
(and
(on a b)
(on c d)
(on d a))
)
)


